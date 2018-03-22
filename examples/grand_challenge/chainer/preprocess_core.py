import json
import datetime
import os
from socket import gethostname
import yaml


def load_options(filepath):
    """Load options as a Python dict from a file.

    Only Json and Yaml files are supported.

    Args:
        filepath(str): Path to a file which contains options.

    Returns:
        dict: Loaded dictionary.

    """
    _, extension = os.path.splitext(filepath)
    if extension == '.json':
        loader = json.load
    elif extension == '.yaml' or extension == '.yml':
        loader = yaml.load
    else:
        raise ValueError('Extension {} is not supported.'.format(extension))

    with open(filepath, 'r') as f:
        return loader(f)


class TimeSingleton(object):
    """Singleton for getting a time

    To use single timestamp for all, we use singleton.

    """
    _instance = None
    _date = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if self._date is None:
            self._date = datetime.datetime.now()

    def __call__(self, fmt='%y.%m.%d_%H.%M.%S'):
        return self._date.strftime(fmt)


def get_time(fmt='%y.%m.%d_%H.%M.%S'):
    """Get timestamp. 

    Args:
        fmt(str): Format to parse the datetime.
    """
    t = TimeSingleton()
    return t(fmt)


def get_npernode(options):
    hostname = gethostname()
    if hostname == 'kfc.r.gsic.titech.ac.jp':
        return 8
    else:
        if options['nodetype'] == 'f_node':
            return 4
        elif options['nodetype'] == 'h_node':
            return 2
        elif options['nodetype'] == 'q_node':
            return 1
        elif options['nodetype'] == 's_gpu':
            return 1
        else:
            raise ValueError('Unsupported nodetype: {}'.format(
                options['nodetype']))


def get_np(options):
    npernode = get_npernode(options)
    return npernode * options['nnodes']

