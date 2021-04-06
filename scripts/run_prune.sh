CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/predcessor/
export OUTPUT_DIR=$CURRENT_DIR/intermedia/
export DATA_DIR=$CURRENT_DIR/processed/

MODE='prune'
BERT_MODEL='bert-base-chinese'

python main.py \
  --model_dir=$MODEL_DIR \
   --bert_model=$BERT_MODEL \
  --output_dir=$OUTPUT_DIR/pytorch_model.bin \
  --trainset=$DATA_DIR/train.txt \
  --validset=$DATA_DIR/dev.txt \
  --batch_size=32 \
  --n_epochs=50 \
  --lr=1e-05 \
  --mode=$MODE \
  --fine_tune_scc \
  --thesues \
  --scc_layer=6 \
  --replacing_rate=0.3 \
  --scheduler_type=linear \
  --scheduler_linear_k=0.006 \
  --steps_for_replacing=500