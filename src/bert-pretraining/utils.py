import tensorflow as tf
import numpy as np
import sys

def set_tensorflow_random_seed(rand_seed):
    random.seed(config['seed'])
	np.random.seed(config['seed'])
	tf.set_random_seed(config['seed'])
