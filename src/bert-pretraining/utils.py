import tensorflow as tf
import numpy as np
import sys
import datetime

def set_tensorflow_random_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)

def get_date_str():
    return '{:%Y-%m-%d}'.format(datetime.date.today())

def ensure_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
