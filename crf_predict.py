# -*- encoding: utf-8 -*-
'''
@File    :   predict.py
@Time    :   2019/12/27 19:20:51
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   BERT-BILSTM-CRF FOR BER
'''
import torch
import time
from utils import tag2idx, idx2tag
from crf import Bert_BiLSTM_CRF
from transformers import BertTokenizer


CRF_MODEL_PATH = './successor-constant'
# BERT_PATH = 'huawei-noah/TinyBERT_6L_zh'
BERT_PATH = 'hfl/chinese-roberta-wwm-ext'


class CRF(object):
    def __init__(self, crf_model, bert_model, device='cpu'):
        self.device = torch.device(device)
        self.model = Bert_BiLSTM_CRF(tag2idx, model_dir=crf_model, bert_thesues=False, fine_tune_scc=False,
                                     device=self.device, hidden_dim=768, scc_layer=3)
        self.model.load_state_dict(torch.load(crf_model+'/pytorch_model.bin', map_location=self.device), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
    def predict(self, text):
        """Using CRF to predict label
        
        Arguments:
            text {str} -- [description]
        """
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        xx = self.tokenizer.convert_tokens_to_ids(tokens)
        xx = torch.tensor(xx).unsqueeze(0).to(self.device)
        _, y_hat = self.model(xx)
        pred_tags = []
        for tag in y_hat.squeeze():
            pred_tags.append(idx2tag[tag.item()])
        return pred_tags, tokens

    def parse(self, tokens, pred_tags):
        """Parse the predict tags to real word
        
        Arguments:
            x {List[str]} -- the origin text
            pred_tags {List[str]} -- predicted tags

        Return:
            entities {List[str]} -- a list of entities
        """
        entities = []
        entity = None
        tag = ''
        for idx, st in enumerate(pred_tags):
            if entity is None:
                if st.startswith('B'):
                    entity = {}
                    entity['start'] = idx
                    tag = st[2:]
                else:
                    continue
            else:
                if st == 'O':
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start']: entity['end']])
                    entities.append((name, tag))
                    entity = None
                    tag = ''
                elif st.startswith('B'):
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start']: entity['end']])
                    entities.append((name, tag))
                    entity = {}
                    entity['start'] = idx
                    tag = st[2:]
                else:
                    continue
        return entities


def get_crf_ners(text):
    pred_tags, tokens = crf.predict(text)
    entities = crf.parse(tokens, pred_tags)
    return entities


if __name__ == "__main__":
    import re

    texts = '罗红霉素和头孢能一起吃吗'

    start = time.time()
    crf = CRF(CRF_MODEL_PATH, BERT_PATH, 'cpu')
    get_crf_ners(text)
    print(f"time:\t{time.time()-start}")
