CURRENT_DIR=`pwd`
export MODEL_DIR="bert-base-chinese"
export OUTPUT_DIR=$CURRENT_DIR/prdecessor/
export DATA_DIR=$CURRENT_DIR/processed/

MODE='prdecessor'
BERT_MODEL='bert-base-chinese'

python main.py \
  --model_dir=$MODEL_DIR \
   --bert_model=$BERT_MODEL \
  --output_dir=$OUTPUT_DIR/pytorch_model.bin \
  --trainset=$DATA_DIR/train.txt \
  --validset=$DATA_DIR/dev.txt \
  --batch_size=32 \
  --n_epochs=10 \
  --lr=1e-05 \
  --mode=$MODE

