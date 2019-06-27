#!/bin/bash

# need make sure conda is installed already
conda create -n bert-pretraining python=3.6

source activate bert-pretraining
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp36-cp36m-linux_x86_64.whl

pushd ../third_party/pytorch-pretrained-BERT
pip install -e .
popd

source deactivate