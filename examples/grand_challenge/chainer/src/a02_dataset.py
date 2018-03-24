from chainer.dataset import dataset_mixin
import chainermn
import numpy as np

import dlframeworks


def get_dataset(args, comm, model):
    if args.optimizer == 'kfac':
        if comm.is_inv_worker:
            return DummyDataset()
        comm = comm.gccomm

    mean = np.load(args.mean)
    if args.loadtype == 'development':
        if comm.rank == 0:
            train = dlframeworks.chainer.datasets.read_pairs(args.train)
            val = dlframeworks.chainer.datasets.read_pairs(args.val)
        else:
            train = None
            val = None
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        val = chainermn.scatter_dataset(val, comm)
        train = dlframeworks.chainer.datasets.CroppingDataset(
            train, args.train_root, mean, model.insize, model.insize)
        val = dlframeworks.chainer.datasets.CroppingDataset(
            val, args.val_root, mean, model.insize, model.insize, False)
    else:
        raise NotImplementedError('Invalid loadtype: {}'.format(args.loadtype))
    return train, val


class DummyDataset(dataset_mixin.DatasetMixin):

    def __len__(self):
        return 0

    def get_example(self, i):
        return None
