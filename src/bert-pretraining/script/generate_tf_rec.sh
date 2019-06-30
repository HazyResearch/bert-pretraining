#!/bin/bash

pushd ../../third_party/bert

BASE_DIR=../../..
EMBED_DATA_DIR=gs://embeddings-data/

names=(
	wiki17
	wiki18
	)
for name in "${names[@]}"; do
    echo "generating for $name"

	python create_pretraining_data.py \
	  --input_file=$BASE_DIR/data/wiki/${name}/wiki_txt/wiki_bert.txt \
	  --output_file=$EMBED_DATA_DIR/bert-wiki/${name}/wiki_tf_rec/full_tf_examples.tfrecord \
	  --vocab_file=$BASE_DIR/data/bert/uncased_L-12_H-768_A-12/vocab.txt \
	  --do_lower_case=True \
	  --max_seq_length=128 \
	  --max_predictions_per_seq=20 \
	  --masked_lm_prob=0.15 \
	  --random_seed=123 \
	  --dupe_factor=5
 done

popd