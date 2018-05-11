from chainermn.communicators import mpi_communicator_base
import numpy as np


class KFACCommunicatorBase(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm, debug=False):
        super(KFACCommunicatorBase, self).__init__(mpi_comm)
        self.debug = debug
        self.is_setup = False

    def setup(self, fisher_blocks):
        if self.is_setup:
            return

        n = len(fisher_blocks)
        is_inv_worker = True if self.rank < n else False
        if self.size > n:
            inv_comm = self.split(int(is_inv_worker), self.rank)
        else:
            inv_comm = self

        indices = np.array_split(np.arange(n), self.size)
        indices = [local_indices.tolist() for local_indices in indices]

        self.is_inv_worker = is_inv_worker
        self.inv_comm = inv_comm
        self.indices = indices
        self.is_setup = True

    def allreduce_grad(self, *args, **kwargs):
        pass

    def reduce_scatterv_grad(self, fisher_blocks, root=0):
        raise NotImplementedError

    def allgatherv_kfgrad(self, fisher_blocks):
        raise NotImplementedError
