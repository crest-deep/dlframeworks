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


class Obj(object):

    def __init__(self, comm_name):
        self.comm_name = comm_name
        self.comm = dlframeworks.chainer.communicators.KFACCommunicator(
            MPI.COMM_WORLD, comm_name)
        self.device = self.comm.intra_rank
        chainer.cuda.get_device(self.device).use()
        self.model = MLP().to_gpu()
        self.optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), self.comm)
        self.optimizer.setup(self.model)

    def __call__(self, x):
        y = self.model(x)
        y.backward()


def generate_random_arrays(shape, n, index):
    arrays = []
    for i in range(n):
        array = cp.ones(shape).astype(cp.float32) * (i + 1)
        arrays.append(array)

    for i in range(n):
        mean_array = cp.zeros(shape).astype(cp.float32)
        mean_array += arrays[i]
    mean_array /= n

    return mean_array, arrays[0], arrays[index]


class KFACCommunicatorTest(unittest.TestCase):

    @parameterized.expand([
        'pure_nccl',
        'hierarchical',
        'two_dimensional',
        'flat',
        'naive',
    ])
    def test_allreduce_grad(self, comm_name):
        obj = Obj(comm_name)
        obj(cp.array([[1, 2]], dtype=cp.float32))

        if not obj.comm.is_grad_worker:
            return

        mean_arrays = {}
        first_arrays = {}
        for name, param in sorted(obj.model.namedparams()):
            shape = param.data.shape
            mean_array, first_array, array = generate_random_arrays(
                shape, obj.comm.gcomm.size, obj.comm.gcomm.rank)
            param.data = array
            param.grad = array
            mean_arrays[name] = mean_array
            first_arrays[name] = first_array

        synced = obj.comm.allreduce_grad(obj.optimizer)

        for name, param in sorted(obj.model.namedparams()):
            if synced:
                cp.testing.assert_array_almost_equal(
                    param.grad, mean_arrays[name])
            else:
                cp.testing.assert_array_almost_equal(
                    param.data, first_arrays[name])

    @parameterized.expand([
        'pure_nccl',
        'hierarchical',
        'two_dimensional',
        'flat',
        'naive',
    ])
    def test_bcast_inv(self, comm_name):
        obj = Obj(comm_name)
        obj(cp.array([[1, 2]], dtype=cp.float32))

        if not obj.comm.is_grad_worker or not obj.comm.is_inv_worker:
            return

        invs = collections.OrderedDict()
        first_arrays = {}
        for name, _ in model.namedparams():
            _, first_array, array = generate_random_arrays(
                (4, 4), obj.comm.gcomm_g.size, obj.comm.gcomm_g.rank)
            invs[name] = [array, array + 1]
            first_arrays[name] = first_array

        self.comm.bcast_inv(invs)

        for name, matrices in invs.items():
            for i, matrix in enumerate(matrices):
                cp.testing.assert_array_almost_equal(
                    matrix, first_array[name] + i)

    @parameterized.expand([
        'pure_nccl',
        'hierarchical',
        'two_dimensional',
        'flat',
        'naive',
    ])
    def test_allreduce_cov(self, comm_name):
        obj = Obj(comm_name)
        obj(cp.array([[1, 2]], dtype=cp.float32))

        if not obj.comm.is_cov_worker:
            return

        covs = []
        mean_arrays = []
        for i in range(3):
            mean_array, _, array = generate_random_arrays(
                (4, 4), obj.comm.ccomm.size, obj.comm.ccomm.rank)
            covs.append(array)
            mean_arrays.append(mean_array)
        
        obj.comm.allreduce_cov(covs)

        for i in range(3):
            cp.testing.assert_array_almost_equal(
                covs[i], mean_arrays[i])

    @parameterized.expand([
        'pure_nccl',
        'hierarchical',
        'two_dimensional',
        'flat',
        'naive',
    ])
    def test_sendrecv_param(self, comm_name):
        obj = Obj(comm_name)
        obj(cp.array([[1, 2]], dtype=cp.float32))

        if not obj.comm.is_grad_master or not obj.comm.is_cov_worker:
            return

        first_arrays = {}
        for name, param in sorted(obj.model.namedparams()):
            shape = param.data.shape
            _, first_array, array = generate_random_arrays(
                shape, 2, obj.comm.is_grad_master)
            param.data = array
            param.grad = array
            first_arrays[name] = first_array

        obj.comm.sendrecv_param(optimizer)

        for name, param in sorted(obj.model.namedparams()):
                cp.testing.assert_array_almost_equal(
                    param.data, first_arrays[name])

#    def test_sendrecv_cov_ema(self):
#        model = MLP()
#        cov_emas = collections.OrderedDict()
#        for name, _ in model.namedparams():
#            if self.comm.is_cov_worker:
#                cov_emas[name] = np.ones((4, 4), dtype=np.float32)
#            else:
#                cov_emas[name] = None
#
#        self.comm.sendrecv_cov_ema(cov_emas)
#
#        if self.comm.is_inv_worker:
#            self.assertTrue(len(cov_emas) == len(list(model.namedparams())),
#                            'expected {} however {}'.format(
#                            len(list(model.namedparams())),
#                            len(cov_emas)))
#            for name, cov_ema in cov_emas.items():
#                self.assertTrue(np.all(cov_ema == 1),
#                                'expected 1, however {}'.format(cov_ema[0]))
#

def main():
    unittest.main()


if __name__ == '__main__':
    main()
