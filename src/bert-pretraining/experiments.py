import json
import glob
import utils

SCRIPT_FOLDER="../../script"

def bert_pretraining_lr_tuning_training():
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_lr_tuning_training"
    lrs = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    BERT_BASE_DIR = "../../data/bert"
    tpu_tmp = 'gcloud compute tpus create tpu-{} --range=10.240.{}.0 --version=1.13 --accelerator-type=v2-8 --network=default &'
    run_tmp = ('python ./third_party/bert/run_pretraining.py \
            --input_file=gs://embeddings-data2/bert-wiki/wiki17/wiki_tf_rec/part_tf_examples_*.tfrecord \
            --output_dir=gs://embeddings-ckpt/bert_pretraining_lr_tuning/pretrain_tuning_lr_{}  \
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
            --tpu_name=tpu-{} 2>&1 | tee output/pretrain_tuning_lr_{}.log &')
    with open(file_name, 'w') as f:
        #for i, lr in enumerate(lrs):
        #    cmd_str = tpu_tmp.format(i, i)
        #    f.write(cmd_str + "\n")
        for i, lr in enumerate(lrs):
            cmd_str = run_tmp.format(lr, lr, i, lr)
            f.write(cmd_str + "\n")

def bert_pretraining_lr_tuning_evaluation():
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_lr_tuning_eval"
    print("cmd in ", file_name)
    lrs = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    BERT_BASE_DIR = "../../data/bert"
    tpu_tmp = 'gcloud compute tpus create tpu-{} --range=10.240.{}.0 --version=1.13 --accelerator-type=v2-8 --network=default &'
    run_tmp = ('python ./third_party/bert/run_pretraining.py \
            --input_file=gs://embeddings-data2/bert-wiki/wiki17/wiki_tf_rec/eval_full_tf_examples.tfrecord \
            --output_dir=gs://embeddings-ckpt/bert_pretraining_lr_tuning/pretrain_tuning_lr_{}_eval  \
            --do_eval=True \
            --bert_config_file=../../data/bert/3_layer_bert_config.json \
            --eval_batch_size=256 \
            --init_checkpoint=gs://embeddings-ckpt/bert_pretraining_lr_tuning/pretrain_tuning_lr_{}/model.ckpt-250000 \
            --max_seq_length=128 \
            --max_predictions_per_seq=20 \
            --use_tpu=True \
            --tpu_name=tpu-0 2>&1 | tee output/pretrain_tuning_lr_{}_eval.log')
    with open(file_name, 'w') as f:
        #for i, lr in enumerate(lrs):
        #   cmd_str = tpu_tmp.format(i, i)
        #   f.write(cmd_str + "\n")
        for i, lr in enumerate(lrs):
            cmd_str = run_tmp.format(lr, lr, i, lr)
            f.write(cmd_str + "\n")

def bert_pretraining_3_seeds_different_size():
    # the optimial learning rate from grid search using 768 dimension is 0.0001
    # generate script to launch tpu
    dims = [192, 384, 768, 1536, 3072]
    for dim in dims:
        with open("../../data/bert/3_layer_dim_{}_bert_config.json".format(dim), "w") as f_out:
            with open("../../data/bert/3_layer_bert_config.json", "r") as f_in:
                for line in f_in.readlines():
                    if "hidden_size" in line:
                        line = line.replace("768", str(dim))
                    f_out.write(line)

    run_tmp = ('python ./third_party/bert/run_pretraining.py --rand_seed={} \
            --input_file=gs://embeddings-data2/bert-wiki/{}/wiki_tf_rec/part_tf_examples_*.tfrecord \
            --output_dir=gs://embeddings-ckpt/bert_pretraining_3_seeds/pretrain_seed_{}_dim_{}_{}  \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=../../data/bert/3_layer_dim_{}_bert_config.json \
            --train_batch_size=256 \
            --max_seq_length=128 \
            --max_predictions_per_seq=20 \
            --num_train_steps=250000 \
            --num_warmup_steps=2500 \
            --learning_rate=0.0001 \
            --use_tpu=True \
            --tpu_name=tpu-{} 2>&1 | tee output/pretrain_seed_{}_dim_{}_{}.log \n')
    tpu_id = 0
    for name in ['wiki17', 'wiki18']:
        for seed in [1,2,3]:
            file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_tpu_{}".format(tpu_id)
            print("cmd saved in ", file_name)
            with open(file_name, "w") as f:
                dim = 192
                cmd = run_tmp.format(seed, name, seed, dim, name, dim, tpu_id, seed, dim, name)
                f.write(cmd)
                dim = 384
                cmd = run_tmp.format(seed, name, seed, dim, name, dim, tpu_id, seed, dim, name)
                f.write(cmd)
                dim = 768
                cmd = run_tmp.format(seed, name, seed, dim, name, dim, tpu_id, seed, dim, name)
                f.write(cmd)
            tpu_id += 1
            file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_tpu_{}".format(tpu_id)
            print("cmd saved in ", file_name)
            with open(file_name, "w") as f:
                dim = 1536
                cmd = run_tmp.format(seed, name, seed, dim, name, dim, tpu_id, seed, dim, name)
                f.write(cmd)
            tpu_id += 1
            file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_tpu_{}".format(tpu_id)
            print("cmd saved in ", file_name)
            with open(file_name, "w") as f:
                dim = 3072
                cmd = run_tmp.format(seed, name, seed, dim, name, dim, tpu_id, seed, dim, name)
                f.write(cmd)
            tpu_id += 1
    # for launch tpu machines
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_tpu_launch"
    with open(file_name, "w") as f:
        for i in range(tpu_id):
            cmd = "gcloud compute tpus create tpu-{} --range=10.240.{}.0 --version=1.13  --accelerator-type=v2-8 --network=default & \n".format(i, i)
            f.write(cmd)
        print("cmd saved in ", file_name)
    # for copy logs into folders
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_copy_logs"
    with open(file_name, "w") as f:
        for name in ['wiki17', 'wiki18']:
            for seed in [1,2,3]:
                for dim in dims:
                    cmd = "gsutil cp output/pretrain_seed_{}_dim_{}_{}.log gs://embeddings-ckpt/bert_pretraining_3_seeds/pretrain_seed_{}_dim_{}_{} \n".format(seed, dim, name, seed, dim, name)
                    f.write(cmd)
        print("cmd saved in ", file_name)
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_copy_configs"
    cmd_tmp = ("gsutil cp ../../data/bert/3_layer_dim_{}_bert_config.json \
                gs://embeddings-ckpt/bert_pretraining_3_seeds/pretrain_seed_{}_dim_{}_{}/bert_config.json")
    with open(file_name, "w") as f:
        for name in ['wiki17', 'wiki18']:
            for seed in [1,2,3]:
                for dim in dims:
                    cmd = cmd_tmp.format(dim, seed, dim, name)
                    f.write(cmd + "\n")
        print("cmd saved in ", file_name)
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_download_bucket"
    # for downloading to local disk
    cmd_tmp = ("gsutil cp -r gs://embeddings-ckpt/bert_pretraining_3_seeds/* \
                ../../results/bert_ckpt/")
    # generate bert pytorch ckpt
    file_name = SCRIPT_FOLDER + "/0701_bert_pretraining_all_seed_trans_to_pytorch"
    folders = glob.glob("../../results/bert_ckpt/*")
    cmd_tmp = ('pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch '
                '{}/model.ckpt-250000 '
                '{}/bert_config.json '
                '{}/pytorch_model.bin')
    with open(file_name, "w") as f:
        for folder in folders:
            cmd = cmd_tmp.format(folder, folder, folder)
            f.write(cmd + "\n")
        print("cmd saved in ", file_name)

def get_feature_path(exp_name, dataset, ckpt_folder, nbit=32):
    exp_path = "../../results/features/{}_{}".format(exp_name, utils.get_date_str())
    ckpt_name = ckpt_folder.split("/")[-1]
    folder = exp_path + "/{}/nbit_{}/{}".format(dataset, nbit, ckpt_name)
    return folder

def get_sentiment_data_path():
    return "./third_party/sentence_classification/data"

def generate_all_sentiment_features_dimensionality():
    ckpt_folders = glob.glob("../../results/bert_ckpt/*")
    # datasets = ['mr', 'sst', 'subj', 'mpqa']
    datasets = ['sst',]
    nbits = [32]
    exp_name = "dimensionality"
    data_path = get_sentiment_data_path()
    script_name = SCRIPT_FOLDER + "/0703_generate_features_for_dimensionality_copy_vocab_file"
    # copy the vocab files to the ckpt folders
    with open(script_name, "w") as f:
        cmd_tmp = "cp ../../data/bert/vocab.txt {}/"
        for ckpt_path in ckpt_folders:
            cmd = cmd_tmp.format(ckpt_path)
            f.write(cmd + "\n")
    print("cmd saved in ", script_name)

    # generate the cmd to generate features
    script_name = SCRIPT_FOLDER + "/0703_generate_features_for_dimensionality"
    with open(script_name, "w") as f:
        cmd_tmp = ('python ./third_party/pytorch-pretrained-BERT/examples/extract_features.py '
                        '--input_file {} '
                        '--output_file {} '
                        '--bert_model {} '
                        '--do_lower_case '
                        '--layer 2 '
                        '--max_seq_length 128 '
                        '--for_sentiment')
        for ckpt_path in ckpt_folders:
            for dataset in datasets:
                for nbit in nbits:
                    output_path = get_feature_path(exp_name, 
                        dataset=dataset, ckpt_folder=ckpt_path, nbit=nbit)
                    for part in ['train', 'test', 'heldout']:
                        input_file = data_path + "/{}.{}.txt".format(dataset, part)
                        output_file = output_path + "/{}.{}.feature.npy".format(dataset, part)
                        cmd = cmd_tmp.format(input_file, output_file, ckpt_path)
                        f.write(cmd + "\n")
        print("cmd saved in ", script_name)

def generate_all_sentiment_features_pytorch_file():
    #for corpus in ['wiki17', 'wiki18']:
    #    ckpt_folders = glob.glob("../../results/bert_ckpt/*{}".format(corpus))
        # generate the cmd to generate features
    ckpt_folders = glob.glob("../../results/bert_ckpt/*")
    datasets = ['mr', 'subj', 'mpqa']
    nbits = [32]
    exp_name = "dimensionality"
    data_path = get_sentiment_data_path()
    # script_name = SCRIPT_FOLDER + "/0706_generate_features_for_dimensionality_pytorch_{}".format(corpus)
    script_name = SCRIPT_FOLDER + "/0706_generate_features_for_dimensionality_pytorch"
    with open(script_name, "w") as f:
        cmd_tmp = ('python ./third_party/pytorch-pretrained-BERT/examples/extract_features.py '
                        '--input_file {} '
                        '--output_file {} '
                        '--bert_model {} '
                        '--do_lower_case '
                        '--layer 2 '
                        '--max_seq_length 128 '
                        '--for_sentiment')
        for ckpt_path in ckpt_folders:
            for dataset in datasets:
                for nbit in nbits:
                    output_path = get_feature_path(exp_name, 
                        dataset=dataset, ckpt_folder=ckpt_path, nbit=nbit)
                    for part in ['train', 'test', 'heldout']:
                        input_file = data_path + "/{}.{}.txt".format(dataset, part)
                        output_file = output_path + "/{}.{}.feature.npz".format(dataset, part)
                        cmd = cmd_tmp.format(input_file, output_file, ckpt_path)
                        f.write(cmd + "\n")
        print("cmd saved in ", script_name)



    
if __name__ == "__main__":
    # bert_pretraining_lr_tuning_training()
    # bert_pretraining_lr_tuning_evaluation()
    # bert_pretraining_3_seeds_different_size()
    # generate_all_sentiment_features_dimensionality()
    generate_all_sentiment_features_pytorch_file()
