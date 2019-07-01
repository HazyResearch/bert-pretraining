import json

SCRIPT_FOLDER="../../script"

def bert_pretraining_lr_tuning_training():
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_lr_tuning_training"
    lrs = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    BERT_BASE_DIR = "../../data/bert"
    tpu_tmp = 'gcloud compute tpus create tpu-{} --range=10.240.{}.0 --version=1.13 --accelerator-type=v2-8 --network=default'
    run_tmp = ('python ../third_party/bert/run_pretraining.py \
            --input_file=gs://embeddings-data2/bert-wiki/wiki17/wiki_tf_rec/part_tf_examples_*.tfrecord \
            --output_dir=gs://embeddings-ckpt/pretrain_tuning_lr_{}  \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=../../data/bert/3_layer_bert_config.json \
            --train_batch_size=256 \
            --max_seq_length=128 \
            --max_predictions_per_seq=20 \
            --num_train_steps=250000 \
            --num_warmup_steps=2500 \
            --learning_rate={} \
            --use_tpu=True \
            --tpu_name=tpu-{} 2>&1 | tee pretrain_tuning_lr_{}.log &')
    with open(file_name, 'w') as f:
        for i, lr in enumerate(lrs):
            cmd_str = tpu_tmp.format(i, i)
        for i, lr in enumerate(lrs):
            cmd_str = run_tmp.format(lr, lr, i, lr)
            f.write(cmd_str + "\n")
    
if __name__ == "__main__":
    bert_pretraining_lr_tuning_training()


