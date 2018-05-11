import datetime
import json
import socket
import shutil
import yaml

import os


def load_ops(path):
    _, ext = os.path.splitext(path)
    if ext == '.json':
        loader = json.load
    elif ext == '.yaml' or ext == '.yml':
        loader = yaml.load
    else:
        raise ValueError

    with open(path, 'r') as f:
        return loader(f)


def get_time(fmt='%y.%m.%d_%H.%M.%S'):
    t = datetime.datetime.now()
    return t.strftime(fmt)


def get_npernode(ops):
    hostname = gethostname()[:3]
    if hostname == 'kfc':
        return 8
    else:
        if ops['nodetype'] == 'f_node':
            return 4
        elif ops['nodetype'] == 'h_node':
            return 2
        elif ops['nodetype'] == 'q_node':
            return 1
        elif ops['nodetype'] == 's_gpu':
            return 1
        else:
            raise ValueError


def get_np(ops):
    npernode = get_npernode(ops)
    return npernode * ops['nnodes']


def copy_files(dst, *args, src='.'):
    ignore = shutil.ignore_patterns(*args)
    shutil.copytree(src, dst, ignore=ignore)
