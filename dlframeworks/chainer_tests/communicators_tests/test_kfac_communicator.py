import chainer
import chainer.links as L
import chainer.functions as F
import chainermn
import collections
from parameterized import parameterized
import mock
import unittest
import numpy as np
import cupy as cp
import random

import dlframeworks


def create_print_mpi(comm):
    import mpi4py.MPI
    rank = comm.rank
    size = comm.size
    host = mpi4py.MPI.Get_processor_name()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def print_mpi(obj, root=None):
        for i in range(size):
            if i == rank:
                if root is not None:
                    if i == root:
                        print(prefix, end='')
                        print(obj)
                else:
                    print(prefix, end='')
                    print(obj)
            comm.Barrier()
    return print_mpi


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
        if not self.comm.is_grad_worker:
            return

        rank = self.comm.gcomm.rank
        size = self.comm.gcomm.size

        for i in range(10):
            roots = {}
            means = {}
            for name, param in sorted(self.model.namedparams()):
                if param is None:
                    continue
                shape = param.data.shape
                array = cp.random.rand(*shape, dtype=cp.float32)
                param.data[:] = array[:]
                array_cpu = chainer.cuda.to_cpu(array)
                roots[name] = self.comm.gcomm.mpi_comm.bcast(array_cpu)

                array = cp.random.rand(*shape, dtype=cp.float32)
                param.grad[:] = array[:]
                array_cpu = chainer.cuda.to_cpu(array)
                means[name] = self.comm.gcomm.mpi_comm.allreduce(array_cpu)
                means[name] /= size

            synced = self.comm.allreduce_grad(self.optimizer)

            for name, param in self.model.namedparams():
                if synced:
                    grad = chainer.cuda.to_cpu(param.grad)
                    np.testing.assert_array_almost_equal(grad, means[name])
                else:
                    data = chainer.cuda.to_cpu(param.data)
                    cp.testing.assert_array_almost_equal(data, roots[name]) 

    def test_bcast_inv(self):
        if self.comm.is_cov_worker:
            return

        n_invs = 4
        print_mpi = create_print_mpi(self.comm.gcomm_g.mpi_comm)

        invs = collections.defaultdict(lambda: [])
        roots = collections.defaultdict(lambda: [])
        for name, param in sorted(self.model.namedparams()):
            for i in range(n_invs):
                shape = (i + 1024, i + 1024)
                if self.comm.is_grad_worker:
                    array = cp.zeros(shape, dtype=cp.float32)
                else:
                    array = cp.random.rand(*shape, dtype=cp.float32)
                invs[name].append(array)
                array_cpu = chainer.cuda.to_cpu(array)
                root_array = self.comm.gcomm_g.mpi_comm.bcast(array_cpu)
                roots[name].append(root_array)

        self.comm.bcast_inv(invs)

        for name, inv in sorted(invs.items()):
            for i, matrix in enumerate(inv):
                matrix_cpu = chainer.cuda.to_cpu(matrix)
                np.testing.assert_array_almost_equal(
                    matrix_cpu, roots[name][i])

    def test_allreduce_cov(self):
        if not self.comm.is_cov_worker:
            return

        n_covs = 4
        size = self.comm.gcomm_g.size
        print_mpi = create_print_mpi(self.comm.ccomm.mpi_comm)

        covs = []
        means = []
        for i in range(n_covs):
            shape = (i + 1, i + 1)
            if self.comm.is_grad_worker:
                array = cp.zeros(shape, dtype=cp.float32)
            else:
                array = cp.random.rand(*shape, dtype=cp.float32)
            covs.append(array)
            array_cpu = chainer.cuda.to_cpu(array)
            mean = self.comm.gcomm_g.mpi_comm.allreduce(array_cpu)
            mean /= size
            means.append(mean)
        print_mpi(means)

        self.comm.allreduce_cov(covs)

        for cov, mean in zip(covs, means):
            cov_cpu = chainer.cuda.to_cpu(cov)
            np.testing.assert_array_almost_equal(
                cov_cpu, mean)


    def test_sendrecv_param(self):
        if not self.comm.is_grad_master and not self.comm.is_cov_worker:
            return

        is_sender = self.comm.is_grad_master
        is_reciever = self.comm.is_cov_worker

        for i in range(10):
            arrays = {}
            for name, param in sorted(self.model.namedparams()):
                if param is None:
                    continue
                if is_sender:
                    shape = param.data.shape
                    array = cp.random.rand(*shape, dtype=cp.float32)
                    param.data[:] = array[:]
                    array_cpu = chainer.cuda.to_cpu(array)
                    self.comm.wcomm.mpi_comm.send(array_cpu, dest=self.comm.cov_worker_rank)
                elif is_reciever:
                    array = self.comm.wcomm.mpi_comm.recv(source=self.comm.grad_master_rank)
                    arrays[name] = array

            self.comm.sendrecv_param(self.optimizer)

            if is_reciever:
                for name, param in sorted(self.model.namedparams()):
                    if param is None:
                        continue
                    data = chainer.cuda.to_cpu(param.data)
                    np.testing.assert_array_almost_equal(
                        data, arrays[name])


    def test_sendrecv_cov_ema(self):
        if not self.comm.is_cov_worker and not self.comm.is_inv_worker:
            return

        n_cov_emas = 4
        is_sender = self.comm.is_cov_worker
        is_reciever = self.comm.is_inv_worker

        cov_emas = collections.defaultdict(lambda: [])
        arrays = collections.defaultdict(lambda: [])
        for name, param in sorted(self.model.namedparams()):
            for i in range(n_cov_emas):
                shape = (i + 1, i + 1)
                if is_reciever:
                    array = self.comm.wcomm.mpi_comm.recv(
                        source=self.comm.cov_worker_rank)
                    arrays[name].append(array)
                else:
                    array = cp.random.rand(*shape, dtype=cp.float32)
                    cov_emas[name].append(array)
                    array_cpu = chainer.cuda.to_cpu(array)
                    self.comm.wcomm.mpi_comm.send(array_cpu, dest=self.comm.inv_worker_rank)

        if is_sender:
            print(arrays)
        self.comm.sendrecv_cov_ema(cov_emas)
        if is_reciever:
            print(cov_emas)

        if is_reciever:
            for cov_ema, array in zip(cov_emas, arrays):
                cov_ema_cpu = chainer.cuda.to_cpu(cov_ema)
                np.testing.assert_array_almost_equal(
                    cov_ema_cpu, array)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
