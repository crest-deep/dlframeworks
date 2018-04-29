from chainermn.communicators import mpi_communicator_base
import numpy as np

from dlframeworks.chainer.utils import create_mpi_print
from dlframeworks.chainer.utils import get_link
from dlframeworks.chainer.utils import get_linknames


class KFACCommunicatorBase(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm, use_nccl=False, dynamic=False, debug=False):
        super(KFACCommunicatorBase, self).__init__(mpi_comm, use_nccl)
        self.dynamic = dynamic
        self.debug = debug
        self.is_setup = False

    def allreduce_grad(self, *args, **kwargs):
        pass

    def setup(self, model):
        if self.is_setup and not self.dynamic:
            return
        rank = self.rank
        size = self.size

        linknames = sorted(get_linknames(model))
        is_worker = True if rank < len(linknames) else False
        invcomm = self
        if size > len(linknames):
            invcomm = self.split(int(is_worker), rank)
        divided_linknames = np.array_split(linknames, self.size)

        self.linknames = linknames
        self.is_worker = is_worker
        self.invcomm = invcomm
        self.divided_linknames = divided_linknames
        self.is_setup = True

    def reduce_scatterv(self, model, covs, root=0):
        raise NotImplementedError

    def reduce_scatterv_extract(self, model, covs):
        linknames = sorted(get_linknames(model))
        ret = {}
        for linkname in linknames:
            ret[linkname] = []
            if linkname in covs.keys():
                for cov in covs[linkname]:
                    ret[linkname].append(cov)
            link = get_link(model, linkname)
            for _, param in sorted(link.namedparams()):
                if param.grad is not None:
                    ret[linkname].append(param.grad)
        return ret

    def reduce_scatterv_get_nelems(self, dictionary):
        nelems = 0
        for _, arrays in sorted(dictionary.items()):
            for array in arrays:
                nelems += array.size
        return nelems

    def reduce_scatterv_debug(self, dictionary, prefix):
        mpi_print = create_mpi_print(self.mpi_comm)
        idx = 0
        for linkname, arrays in sorted(dictionary.items()):
            for array in arrays:
                mpi_print('{} REDUCE_SCATTERV IDX {} MEAN {}'
                          .format(prefix, idx, array.mean()))
                idx += 1

    def allgatherv(self, model):
        raise NotImplementedError

    def allgatherv_get_nelems(self, model):
        nelems = 0
        for _, param in sorted(model.namedparams()):
            if param.kfgrad is None:
                continue
            nelems += param.kfgrad.size
        return nelems

    def allgatherv_debug(self, model, prefix):
        mpi_print = create_mpi_print(self.mpi_comm)
        idx = 0
        for _, param in sorted(model.namedparams()):
            mpi_print('{} ALLGATHERV IDX {} KFGRAD_MEAN {}'.format(
                prefix, idx, param.kfgrad.mean()))
            idx += 1
