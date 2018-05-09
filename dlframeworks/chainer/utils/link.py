import numpy as np


def get_linknames(model):
    linknames = set()
    for paramname, _ in model.namedparams():
        linkname = paramname[:paramname.rfind('/')]
        # TODO(Yohei):
        if 'bn' in linkname:
            continue
        linknames.add(linkname)
    return list(linknames)


def get_link(model, name):
    for linkname, link in model.namedlinks():
        if linkname == name:
            return link


def get_param(model, name):
    for paramname, param in model.namedparams():
        if paramname == name:
            return param


def get_divided_linknames(model, size):
    linknames = sorted(get_linknames(model))
    return np.array_split(linknames, size)
