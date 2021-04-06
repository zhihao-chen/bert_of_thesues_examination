# -*- encoding: utf-8 -*-
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
import re
from transformers import get_linear_schedule_with_warmup, BertTokenizer


from crf import Bert_BiLSTM_CRF
from utils import NerDataset, pad, tag2idx, idx2tag
from replacement_scheduler import ConstantReplacementScheduler, LinearReplacementScheduler


def train(model, iterator, optimizer, device, replacement, mode, scheduler):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        x = x.to(device)
        y = y.to(device)
        _y = y # for monitoring
        model.zero_grad()
        loss = model.neg_log_likelihood(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        loss.backward()

        optimizer.step()
        scheduler.step()
        if mode == 'prune':
            replacement.step()

        if i == 0:
            print("=====sanity check======")
            #print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, f, device):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)

            _, y_hat = model(x)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array([tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + "%.5f" % f1
    with open(final, 'w', encoding='utf-8') as fout:
        result = open("temp", "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.5f"%precision)
    print("recall=%.5f"%recall)
    print("f1=%.5f"%f1)
    return precision, recall, f1


def save_model(state_dict, save_path):
    pattern = re.compile(r'bert\.encoder\.layer')
    new_state_dict = deepcopy(state_dict)
    for key in state_dict:
        if pattern.search(key):
            new_state_dict.pop(key)
    torch.save(new_state_dict, save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    # parser.add_argument("--trainset", type=str, default="processed/processed_training_bio.txt")
    # parser.add_argument("--validset", type=str, default="processed/processed_dev_bio.txt")
    parser.add_argument("--trainset", type=str, required=True)
    parser.add_argument("--validset", type=str, required=True)
    # parser.add_argument("--model_dir", type=str, default='huawei-noah/TinyBERT_6L_zh')
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument('--bert_model', type=str, required=True)
    parser.add_argument("--scc_layer", type=int, default=3)
    parser.add_argument("--fine_tune_scc", type=bool, action='store_true')
    parser.add_argument('--thesues', type=bool, action='store_true')

    parser.add_argument("--replacing_rate", type=float, default=0.3,
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    parser.add_argument("--scheduler_type", default='linear', choices=['none', 'linear'], help="Scheduler function.")
    parser.add_argument("--scheduler_linear_k", default=0.0006, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--steps_for_replacing", default=500, type=int,
                        help="Steps before entering successor fine_tuning (only useful for constant replacing)")
    parser.add_argument('--mode', default='s', help='process, prune, successor')
    # parser.add_argument('--output_dir', default='/data/chenzhihao/bert_bilstm_crf/successor-linear/pytorch_model.bin')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--warmup_steps", default=0, type=int)

    hp = parser.parse_args()

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained(hp.bert_model)

    model = Bert_BiLSTM_CRF(tag2idx, model_dir=hp.model_dir, bert_thesues=hp.thesues, fine_tune_scc=hp.fine_tune_scc,
                            device=device, hidden_dim=768, scc_layer=hp.scc_layer)
    if hp.mode in ['prune', 'successor']:
        # scc_n_layer = model.bert.encoder.scc_n_layer
        # model.bert.encoder.scc_layer = nn.ModuleList(
        #     [deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
        model.load_state_dict(torch.load(hp.model_dir+'/pytorch_model.bin', map_location=torch.device(device)), strict=False)

    model.to(device)
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset, tokenizer)
    eval_dataset = NerDataset(hp.validset, tokenizer)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    t_total = len(train_iter)
    no_decay = ['bias', 'LayerNorm.weight']
    if hp.mode in ["processor", "successor"]:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    elif hp.mode == "prune":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=hp.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=hp.warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Replace rate scheduler
    replacing_rate_scheduler = None
    if hp.mode == 'prune':
        if hp.scheduler_type == 'none':
            replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                    replacing_rate=hp.replacing_rate,
                                                                    replacing_steps=hp.steps_for_replacing)
        elif hp.scheduler_type == 'linear':
            replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                  base_replacing_rate=hp.replacing_rate,
                                                                  k=hp.scheduler_linear_k)

    best_f1 = 0.0
    print('Start Train...,')
    for epoch in range(1, hp.n_epochs+1):  # 每个epoch对dev集进行测试

        train(model, train_iter, optimizer, criterion, torch.device(device), replacing_rate_scheduler, hp.mode, scheduler)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname, torch.device(device))

        if f1 > best_f1:
            best_f1 = f1
            if hp.mode == 'prune':
                save_model(model.state_dict(), hp.output_dir)
            torch.save(model.state_dict(), hp.output_dir)
    print(f"best f1: {best_f1}. weights were saved to {hp.output_dir}")

