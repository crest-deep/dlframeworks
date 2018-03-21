import chainer
import chainer.links as L
import chainer.functions as F
from parameterized import parameterized
import mock
import unittest
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


def setup_comm(comm, setting):
    comm.is_inv_worker = setting[0]
    comm.is_cov_worker = setting[1]
    comm.is_grad_worker = setting[2]
    comm.is_grad_master = setting[3]


class KFACTest(unittest.TestCase):

    def setUp(self):
        self.model = MLP()
        self.comm = mock.MagicMock()
        self.optimizer = dlframeworks.chainer.optimizers.KFAC(
            self.comm, inv_freq=5)
        self.optimizer.setup(self.model)

    @parameterized.expand([
        ([1, 0, 0, 0], [0, 0, 1], 'testing inv_worker'),
        ([0, 1, 0, 0], [0, 1, 0], 'testing cov_worker'),
        ([0, 0, 1, 0], [1, 0, 0], 'testing grad_worker'),
        ([0, 0, 1, 1], [1, 0, 0], 'testing grad_master')
    ])
    def test_update(self, setting, answers, message):
        self.optimizer.grad_update = mock.MagicMock()
        self.optimizer.cov_ema_update = mock.MagicMock()
        self.optimizer.inv_update = mock.MagicMock()
        setup_comm(self.comm, setting)

        self.optimizer.update()

        print(message, end='...')
        update_attrs = ['grad_update', 'cov_ema_update', 'inv_update']
        for answer, update_attr in zip(answers, update_attrs):
            x = getattr(self.optimizer, update_attr)
            self.assertEqual(x.call_count, answer)
        print('pass')

    @parameterized.expand([
        np.array([[[1, 2]]], dtype=np.float32)
    ])
    def test_grad_update(self, x):
        n = 20
        freq = self.optimizer.hyperparam.inv_freq
        for i in range(n):
            self.optimizer.grad_update(self.model, x)

        cnt = self.comm.sendrecv_param.call_count
        self.assertEqual(cnt, n // freq,
                         'sendrecv_param() must be called {} times'.format(
                            n // freq))
        cnt = self.comm.allreduce_grad.call_count
        self.assertEqual(cnt, n,
                         'allreduce_grad() must be called {} times'.format(
                             n))
        cnt = self.comm.bcast_inv.call_count
        self.assertEqual(cnt, n // freq - 1,
                         'bcast_inv() must be called {} times'.format(
                            n // freq - 1))

    @parameterized.expand([
        np.array([[[1, 2]]], dtype=np.float32)
    ])
    def test_cov_ema_update(self, x):
        n = 20
        for i in range(n):
            self.optimizer.cov_ema_update(self.model, x)

        cnt = self.comm.sendrecv_param.call_count
        self.assertEqual(cnt, n,
                         'sendrecv_param() must be called {} times'.format(n))
        cnt = self.comm.sendrecv_cov_ema.call_count
        self.assertEqual(cnt, n,
                         'sendrecv_param() must be called {} times'.format(n))

    def test_inv_update(self):
        n = 20
        for i in range(20):
            self.optimizer.inv_update()

        cnt = self.comm.sendrecv_cov_ema.call_count
        self.assertEqual(cnt, n,
                         'sendrecv_cov_ema() must be called {} times'.format(n))
        cnt = self.comm.bcast_inv.call_count
        self.assertEqual(cnt, n,
                         'bcast_inv() must be called {} times'.format(n))

def main():
    unittest.main()


if __name__ == '__main__':
    main()
