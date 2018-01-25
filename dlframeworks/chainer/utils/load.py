import json
import numpy
import os
import re


def get_snapshots(path, pattern='^snapshot_iter_[1-9][0-9]*$',
                  key=lambda x: os.stat(x).st_mtime):
    filepaths = []
    filenames = os.listdir(path)
    for filename in filenames:
        if os.path.isfile(os.path.join(path, filename)):
            m = re.search(pattern, filename)
            if m is not None:
                filename = os.path.join(path, filename)
                filename = os.path.abspath(filename)
                filepaths.append(filename)
    filepaths.sort(key=key)
    return filepaths


def load_snapshot(path):
    obj = {}
    with numpy.load(path) as f:
        for k in f.files:
            obj[k] = f[k]
    return obj


def load_log(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def extract_array(data, name, root='updater/model:main'):
    for k, v in data.items():
        pattern = os.path.join(root, name)
        pattern = re.sub(r'^/', '', pattern)
        pattern = re.sub(r'/$', '', pattern)
        m = re.search(pattern, k)
        if m is not None:
            return v
    raise ValueError('{}: No such named value in data.'.format(name))
