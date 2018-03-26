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


class KFACUpdateTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(KFACUpdateTest, self).__init__(*args, **kwargs)
        self.comm = dlframeworks.chainer.communicators.KFACCommunicator(
            'naive', debug=True)

    def setUp(self):
        self.inv_freq = 5
        self.model = MLP()
        self.optimizer = dlframeworks.chainer.optimizers.KFAC(
            self.comm, inv_freq=self.inv_freq)
        self.optimizer.setup(self.model)

    @parameterized.expand([
        np.array([[[1, 4]]], dtype=np.float32),
    ])
    def test_update(self, x):

        n_updates_grad = 20
        n_updates_cov = n_updates_grad // self.inv_freq
        n_updates_inv = n_updates_cov

        if self.comm.is_grad_worker:
            for i in range(n_updates_grad):
                self.optimizer.update(self.model, x)

        if self.comm.is_cov_worker:
            for i in range(n_updates_cov):
                self.optimizer.update(self.model, x)

        if self.comm.is_inv_worker:
            for i in range(n_updates_inv):
                self.optimizer.update(self.model, x)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
