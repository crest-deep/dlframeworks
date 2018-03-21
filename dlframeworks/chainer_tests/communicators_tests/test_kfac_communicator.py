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

    def setUp(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.comm = dlframeworks.chainer.communicators.KFACCommunicator(
            self.mpi_comm, 'naive')

    def test_allreduce_grad(self):
        model = MLP()
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), self.comm)
        optimizer.setup(model)
        y = model(np.array([[1, 2]], dtype=np.float32))
        y.backward()

        for i in range(2):
            if self.comm.gcomm.rank == 0:
                for param in model.params():
                    shape = param.data.shape
                    param.data = np.ones(shape, dtype=np.float32)
                    param.grad = np.ones(shape, dtype=np.float32)
            else:
                for param in model.params():
                    shape = param.data.shape
                    param.data = np.zeros(shape, dtype=np.float32)
                    param.grad = np.ones(shape, dtype=np.float32)

            synced = self.comm.allreduce_grad(optimizer)

            if self.comm.is_grad_worker:
                for param in model.params():
                    shape = param.data.shape
                    if synced:
                        self.assertTrue(np.all(param.grad == 1),
                                        'expected 1, however {}'.format(param.grad[0]))
                    else:
                        self.assertTrue(np.all(param.data == 1),
                                        'expected 1, however {}'.format(param.data[0]))

    def test_bcast_inv(self):
        model = MLP()
        invs = collections.OrderedDict()
        for name, _ in model.namedparams():
            if self.comm.gcomm_g.rank == 0:
                invs[name] = np.ones((4, 4), dtype=np.float32)
            else:
                invs[name] = np.zeros((4, 4), dtype=np.float32)

        self.comm.bcast_inv(invs)

        if self.comm.is_grad_worker:
            for name, inv in invs.items():
                self.assertTrue(np.all(inv == 1),
                                'expected 1, however {}'.format(inv[0]))

    def test_allreduce_cov(self):
        covs = [np.ones((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32)]
        
        self.comm.allreduce_cov(covs)

        if self.comm.is_cov_worker:
            for cov in covs:
                self.assertTrue(np.all(cov == 1 / self.comm.ccomm.size),
                                'expected {} however {}'.format(
                                    1 / self.comm.ccomm.size, cov[0]))

    def test_sendrecv_param(self):
        model = MLP()
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), self.comm)
        optimizer.setup(model)
        y = model(np.array([[1, 2]], dtype=np.float32))
        y.backward()

        if self.comm.is_grad_master:
            for param in model.params():
                shape = param.data.shape
                param.data = np.ones(shape, dtype=np.float32)
        else:
            for param in model.params():
                shape = param.data.shape
                param.data = np.zeros(shape, dtype=np.float32)

        synced = self.comm.sendrecv_param(optimizer)

        if self.comm.is_cov_worker or self.comm.is_grad_master:
            self.assertTrue(np.all(param.data == 1),
                            'expected 1, however {}'.format(param.data[0]))
        else:
            self.assertTrue(np.all(param.data == 0),
                            'expected 0, however {}'.format(param.data[0]))

    def test_sendrecv_cov_ema(self):
        model = MLP()
        cov_emas = collections.OrderedDict()
        for name, _ in model.namedparams():
            if self.comm.is_cov_worker:
                cov_emas[name] = np.ones((4, 4), dtype=np.float32)
            else:
                cov_emas[name] = None

        self.comm.sendrecv_cov_ema(cov_emas)

        if self.comm.is_inv_worker:
            self.assertTrue(len(cov_emas) == len(list(model.namedparams())),
                            'expected {} however {}'.format(
                            len(list(model.namedparams())),
                            len(cov_emas)))
            for name, cov_ema in cov_emas.items():
                self.assertTrue(np.all(cov_ema == 1),
                                'expected 1, however {}'.format(cov_ema[0]))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
