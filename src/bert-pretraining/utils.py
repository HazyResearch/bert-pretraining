import tensorflow as tf
import numpy as np
import sys, os
import datetime
import logging
import pathlib
import json

def set_tensorflow_random_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)

def get_date_str():
    return '{:%Y-%m-%d}'.format(datetime.date.today())

def ensure_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def save_to_json(dict_to_write, path):
    with open(path, 'w') as f: json.dump(dict_to_write, f, indent=2)

def init_logging(path):
    """Initialize logfile to be used for experiment."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(path, "run.log"),"w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def non_default_args(parser, args):
    non_default = []
    for action in parser._actions:
        default = action.default
        key = action.dest
        if key in args:
            val = args[key]
            if val != default:
                non_default.append((key,val))
    assert len(non_default) > 0, 'There must be a non-default arg.'
    return non_default

def get_arg_str(parser, args):
    # args needs to be a dict
    for key,val in non_default_args(parser, args):
        if key not in to_skip:
            runname += '{},{}_'.format(key,val)