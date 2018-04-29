from chainermn import nccl

from dlframeworks.chainer.communicators.kfac_communicators \
    import kfac_communicator_base


class PureNCCLCommunicator(kfac_communicator_base.KFACCommunicatorBase):

    def __init__(self, mpi_comm, dynamic=False, debug=False):
        super(PureNCCLCommunicator, self).__init__(
            mpi_comm, True, dynamic, debug)
        if nccl.get_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')
        self._init_ranks()
