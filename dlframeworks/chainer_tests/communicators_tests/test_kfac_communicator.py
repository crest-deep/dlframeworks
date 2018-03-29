import chainer
import chainer.links as L
import chainer.functions as F
import chainermn
import collections
from parameterized import parameterized
import mock
import unittest
from mpi4py import MPI
import numpy as np
import cupy as cp

import dlframeworks


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(2, 3),
            l2=L.Linear(3, 2),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return F.sum(self.l2(h))  # output MUST be 1 dim


class KFACCommunicatorTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(KFACCommunicatorTest, self).__init__(*args, **kwargs)
        self.comm = dlframeworks.chainer.communicators.KFACCommunicator()
        chainer.cuda.get_device_from_id(self.comm.wcomm.intra_rank).use()
        self.model = MLP()
        x = np.array([[1, 2]], dtype=np.float32)
        self.model(x)
        self.model.to_gpu()
        self.optimizer = dlframeworks.chainer.optimizers.KFAC()
        self.optimizer.setup(self.model)

    def test_allreduce_grad(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
