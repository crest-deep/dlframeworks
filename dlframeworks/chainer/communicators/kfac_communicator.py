import chainer
import chainermn
from mpi4py import MPI
import numpy as np


def _create_print_mpi(comm):
    rank = comm.rank
    size = comm.size
    host = MPI.Get_processor_name()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def print_mpi(obj):
        for i in range(size):
            if i == rank:
                print(prefix, end='')
                print(obj)
            comm.Barrier()
    return print_mpi


class DummyLink(object):
    """A dummy link that overrides `namedparams` method"""

    def __init__(self, data):
        self._params = {}
        self._params['/'] = DummyParameter(data)

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    @property
    def data(self):
        return self._params['/'].data


class DummyParameter(object):
    """A dummy link that overrides `grad` method"""

    def __init__(self, data):
        self._data = [data]

    @property
    def data(self):
        return self._data[0]

    @data.setter
    def data(self, data):
        self._data[0] = data

    @property
    def grad(self):
        return self._data[0]

    @grad.setter
    def grad(self, data):
        self._data[0] = data


class KFACCommunicator(object):
    """KFAC communicator

    Assuming that more than 4 processes (GPUs) exist on 1 node. If you want to
    test on local machine run with more than 4 processes.

    Args:
        mpi_comm: MPI4py communicator
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, ``pure_nccl``, or
          ``single_node``)
        npergroup (int): Number of nodes per group.
        debug (bool): Print debug message or not.
    """

    def __init__(self, mpi_comm, communicator_name='hierarchical', npergroup=1,
                 debug=False):
        if debug:
            print_mpi = _create_print_mpi(mpi_comm)

        # World communicator which all processes contains
        wcomm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)

        if wcomm.size < 3:
            raise ValueError('Size of KFACCommunicator must be largaer than 2')
        if npergroup < 1 or not isinstance(npergroup, int):
            raise ValueError('Number of nodes per group must positive int')

        n_group = wcomm.inter_size // npergroup
        if n_group == 0:
            n_group = 1
        group_lst = np.array_split(np.arange(wcomm.size), n_group)
        is_cov_worker = 0
        is_inv_worker = 0
        is_grad_worker = 0
        is_grad_master = 0
        for i, group in enumerate(group_lst):
            if wcomm.inter_rank in group:
                group_id = i
            if wcomm.inter_rank == group[0]:
                if wcomm.intra_rank == 0:
                    # Inverse worker
                    is_inv_worker = 1
                elif wcomm.intra_rank == 1:
                    # Covariance worker
                    is_cov_worker = 1
                elif wcomm.intra_rank == 2:
                    # Gradient master
                    is_grad_worker = 1
                    is_grad_master = 1
                else:
                    # Gradient worker
                    is_grad_worker = 1

        # Communicator for all covariance worker
        ccomm = wcomm.split(color=is_cov_worker, key=wcomm.rank)

        # Communicator for all gradient worker
        gcomm = wcomm.split(color=is_grad_worker, key=wcomm.rank)

        # Communicator for inverse worker and gradient worker PER group
        # color is different per group
        color = group_id * (is_inv_worker & is_grad_worker)
        # key is 0 for inverse worker and 1,2,... for gradient worker
        key = 0 if is_inv_worker else wcomm.rank + 1
        gcomm_g = wcomm.split(color=color, key=key)

        send_buf = [is_inv_worker, is_cov_worker, is_grad_worker,
                    is_grad_master, wcomm.rank, wcomm.size, group_id]
        # ======== COMMUNICATION ========
        # get all flags from all processes
        recv_buf = wcomm.mpi_comm.allgather(send_buf)
        for flags in recv_buf:
            if group_id == flags[6]:
                if flags[0] == 1:
                    # Rank of inv_worker in wcomm of THIS group
                    inv_worker_rank = flags[4]
                elif flags[1] == 1:
                    # Rank of cov_worker in wcomm of THIS group
                    cov_worker_rank = flags[4]
                elif flags[3] == 1:
                    # Rank of grad_master in wcomm of THIS group
                    grad_master_rank = flags[4]

        if debug:
            print_mpi('[{}{}{}{}{}{}:{}]'.format(
                is_inv_worker, is_cov_worker, is_grad_worker, is_grad_master,
                wcomm.rank, wcomm.size, group_id))

        super(KFACCommunicator, self).__setattr__(
            'wcomm', wcomm)
        super(KFACCommunicator, self).__setattr__(
            'ccomm', ccomm)
        super(KFACCommunicator, self).__setattr__(
            'gcomm', gcomm)
        super(KFACCommunicator, self).__setattr__(
            'gcomm_g', gcomm_g)
        super(KFACCommunicator, self).__setattr__(
            'is_inv_worker', is_inv_worker)
        super(KFACCommunicator, self).__setattr__(
            'is_cov_worker', is_cov_worker)
        super(KFACCommunicator, self).__setattr__(
            'is_grad_worker', is_grad_worker)
        super(KFACCommunicator, self).__setattr__(
            'is_grad_master', is_grad_master)
        super(KFACCommunicator, self).__setattr__(
            'group_id', group_id)
        super(KFACCommunicator, self).__setattr__(
            'inv_worker_rank', inv_worker_rank)
        super(KFACCommunicator, self).__setattr__(
            'cov_worker_rank', cov_worker_rank)
        super(KFACCommunicator, self).__setattr__(
            'grad_master_rank', grad_master_rank)

    def __getattr__(self, name):
        return getattr(self.gcomm, name)

    def __setattr__(self, name, value):
        setattr(self.gcomm, name, value)

    def allreduce_grad(self, optimizer):
        """Allreduce gradients calculated by backprop

        Args:
            optimizer (chainer.Optimizer): KFAC optimizer.
        """
        # If optimizer is a link object, then call original Allreduce
        if isinstance(optimizer, chainer.Link):
            self.gcomm.allreduce_grad(optimizer)
            return True
        target = optimizer.target
        if _is_changed(optimizer):
            self.gcomm.broadcast_data(target)
            return False
        else:
            self.gcomm.allreduce_grad(target)
            return True

    def bcast_inv(self, invs):
        """Broadcast inverse matrices

        Args:
            invs (OrderedDict(str, numpy.array)): Send buffer or recieve
                buffer of inverse matrices.
        """

        for linkname, matrix in invs.items():
            matrix_link = DummyLink(matrix)
            self.gcomm_g.broadcast_data(matrix_link)
            invs[linkname] = matrix_link.data

    def allreduce_cov(self, covs):
        """Allreduce covariance matrices

        Args:
            covs (list(numpy.array)): Send buffer or recv buffer of
                covariance matrices.
        """
        for i, matrix in enumerate(covs):
            matrix_link = DummyLink(matrix)
            self.ccomm.allreduce_grad(matrix_link)
            covs[i] = matrix_link.data

    def sendrecv_param(self, optimizer):
        """Send or recieve parameters

        Sender is gradient master and reciever is covariance worker.

        Args:
            optimizer (chainer.Optimizer): KFAC optimizer.
        """
        is_sender = self.is_grad_master
        is_reciever = self.is_cov_worker

        if is_sender:
            for _, param in sorted(optimizer.target.namedparams()):
                self.wcomm.send(param.data, self.cov_worker_rank, 0)
        elif is_reciever:
            for linkname, param in sorted(optimizer.target.namedparams()):
                param.data = self.wcomm.recv(self.grad_master_rank, 0)

    def sendrecv_cov_ema(self, cov_emas):
        """Send or recieve covariances EMA

        Sender is covariance worker and reciever is inverse worker.

        Args:
            cov_emas (OrderedDict(str, numpy/cupy.array)): Send buffer or
                recieve buffer of covariances EMA.
        """
        is_sender = self.is_cov_worker
        is_reciever = self.is_inv_worker

        if is_sender:
            for _, data in cov_emas.items():
                self.wcomm.send(data, self.inv_worker_rank, 0)
        elif is_reciever:
            for linkname, data in cov_emas.items():
                cov_emas[linkname] = self.wcomm.recv(self.cov_worker_rank, 0)


def _is_changed(optimizer):
    target = optimizer.target
    previous_params = optimizer.target_params
    optimizer.target_params = [(name, param.data is not None)
                               for name, param in sorted(target.namedparams())]
    if len(previous_params) != len(optimizer.target_params):
        return True
    for param1, param2 in zip(optimizer.target_params, previous_params):
        if (param1[0] != param2[0]) or (param1[1] != param2[1]):
            return True
    return False


if __name__ == '__main__':
    comm = KFACCommunicator(MPI.COMM_WORLD, communicator_name='naive',
                            debug=True)
