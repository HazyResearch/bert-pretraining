import tensorflow as tf
import numpy as np
import sys

def set_tensorflow_random_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)
