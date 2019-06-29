#!/bin/bash

#mkdir -p ../../../data/bert 
#pushd ../../../data/bert
#wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
##sudo apt-get install zip
#unzip uncased_L-12_H-768_A-12.zip
#popd

export BERT_BASE_DIR=~/bert-pretraining/data/bert/uncased_L-12_H-768_A-12
#export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12

#python ../third_party/bert/create_pretraining_data.py \
#  --input_file=../third_party/bert/sample_text.txt \
#  --output_file=gs://embeddings-data/tf_examples.tfrecord \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --do_lower_case=True \
#  --max_seq_length=128 \
#  --max_predictions_per_seq=20 \
#  --masked_lm_prob=0.15 \
#  --random_seed=12345 \
#  --dupe_factor=5

python ../third_party/bert/run_pretraining.py \
  --input_file=gs://embeddings-data/tf_examples.tfrecord \
  --output_dir=gs://embeddings-ckpt/pipeline-test  \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=demo-tpu
  # --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt

#export BERT_BASE_DIR_NEW=/tmp/pretraining_output
#pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
#  $BERT_BASE_DIR_NEW/model.ckpt-20 \
#  $BERT_BASE_DIR/bert_config.json \
#  $BERT_BASE_DIR_NEW/pytorch_model.bin

# pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
#   $BERT_BASE_DIR/bert_model.ckpt \
#   $BERT_BASE_DIR/bert_config.json \
#   $BERT_BASE_DIR/pytorch_model.bin


# use huggingface bert notebook to verify the transformation is correct.
# use notebooks/Comparing-TF-and-PT-models_tf_pytorch_pipeline_test.ipynb


