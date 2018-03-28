import chainer
from chainer.backends import cuda
import chainermn
import numpy as np


def _create_print_mpi(comm):
    import mpi4py.MPI
    rank = comm.rank
    size = comm.size
    host = mpi4py.MPI.Get_processor_name()
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
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, ``pure_nccl``, or
          ``single_node``)
        mpi_comm: MPI4py communicator
        npergroup (int): Number of nodes per group.
        debug (bool): Print debug message or not.
    """

    def __init__(self, communicator_name='hierarchical', mpi_comm=None,
                 npergroup=1, debug=False):
        if mpi_comm is None:
            import mpi4py.MPI
            mpi_comm = mpi4py.MPI.COMM_WORLD

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
        group_lst = np.array_split(np.arange(wcomm.inter_size), n_group)
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
        if not (is_inv_worker | is_cov_worker | is_grad_worker):
            # Gradient worker
            is_grad_worker = 1

        # Communicator for all covariance workers
        ccomm = wcomm.split(color=is_cov_worker, key=wcomm.rank)

        # Communicator for all gradient workers
        gcomm = wcomm.split(color=is_grad_worker, key=wcomm.rank)

        # Communicator for inverse worker and gradient master PER group
        color = (group_id + 1) * (is_inv_worker | is_grad_worker)
        key = 0 if is_inv_worker else wcomm.rank + 1
        gcomm_g = wcomm.split(color=color, key=key)

        # Communicator for all gradient workers and covariance workers
        color = is_cov_worker | is_grad_worker
        gccomm = wcomm.split(color=color, key=wcomm.rank)

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
            print_mpi('[{}{}{}{}:{}]'.format(
                is_inv_worker,
                is_cov_worker,
                is_grad_worker,
                is_grad_master,
                group_id))
            print_mpi('inv_worker_rank: {}'.format(inv_worker_rank))
            print_mpi('cov_worker_rank: {}'.format(cov_worker_rank))

        super(KFACCommunicator, self).__setattr__(
            'wcomm', wcomm)
        super(KFACCommunicator, self).__setattr__(
            'ccomm', ccomm)
        super(KFACCommunicator, self).__setattr__(
            'gcomm', gcomm)
        super(KFACCommunicator, self).__setattr__(
            'gcomm_g', gcomm_g)
        super(KFACCommunicator, self).__setattr__(
            'gccomm', gccomm)
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
        if not self.is_grad_worker:
            return True
        if isinstance(optimizer, chainer.Link):
            self.gcomm.allreduce_grad(optimizer)
            return True
        if _is_changed(optimizer):
            self.gcomm.broadcast_data(optimizer.target)
            return False
        else:
            self.gcomm.allreduce_grad(optimizer.target)
            return True

    def bcast_inv(self, invs):
        """Broadcast inverse matrices

        Inverse worker sends A^-1 and G^-1 to all gradient workers.

        Args:
            invs (OrderedDict(str, list(numpy/cupy.array))): Send buffer or
                recieve buffer of inverse matrices.
        """
        if not self.is_inv_worker and not self.is_grad_worker:
            return
        print('bcast_inv...', len(invs), self.wcomm.rank)
        for linkname, matrices in sorted(invs.items()):
            for i, matrix in enumerate(matrices):
                matrix_link = DummyLink(matrix)
                self.gcomm_g.broadcast_data(matrix_link)
                invs[linkname][i] = matrix_link.data
        print('bcast_inv... done', len(invs), self.wcomm.rank)

    def allreduce_cov(self, covs):
        """Allreduce covariance matrices

        Args:
            covs (list(numpy/cupy.array)): Send buffer or recv buffer of
                covariance matrices.
        """
        if not self.is_cov_worker:
            return
        for i, matrix in enumerate(covs):
            matrix_link = DummyLink(matrix)
            self.ccomm.allreduce_grad(matrix_link)
            covs[i] = matrix_link.data
            del matrix_link

    def sendrecv_param(self, optimizer):
        """Send or recieve parameters

        Sender is gradient master and reciever is covariance worker.

        Args:
            optimizer (chainer.Optimizer): KFAC optimizer.
        """
        is_sender = self.is_grad_master
        is_reciever = self.is_cov_worker

        if is_sender:
            print('sendrecv_param', self.wcomm.rank)
            for name, param in sorted(optimizer.target.namedparams()):
                data = param.data
                data = chainer.cuda.to_cpu(data).astype(np.float32)
                self.wcomm.send(data, self.cov_worker_rank, 0)
            print('sendrecv_param done', self.wcomm.rank)
        elif is_reciever:
            print('sendrecv_param', self.wcomm.rank)
            for name, param in sorted(optimizer.target.namedparams()):
                data = self.wcomm.recv(self.grad_master_rank, 0)
                with cuda.get_device_from_array(param.data) as dev:
                    if dev.id < 0:
                        param.data[:] = data
                    else:
                        param.data[:] = chainer.cuda.to_gpu(data)
            print('sendrecv_param done', self.wcomm.rank)

    def sendrecv_cov_ema(self, cov_emas):
        """Send or recieve covariance EMAs

        Sender is covariance worker and reciever is inverse worker.

        Args:
            cov_emas (dict(str, list(numpy/cupy.array))): Send buffer or
                recieve buffer of covariance EMAs.
        """
        is_sender = self.is_cov_worker
        is_reciever = self.is_inv_worker

        if is_sender:
            print('sendrecv_cov_ema', len(cov_emas), self.wcomm.rank)
            for _, matrices in sorted(cov_emas.items()):
                for matrix in matrices:
                    matrix = chainer.cuda.to_cpu(matrix).astype(np.float32)
                    self.wcomm.send(matrix, self.inv_worker_rank, 0)
            print('sendrecv_cov_ema done', len(cov_emas), self.wcomm.rank)
        elif is_reciever:
            print('sendrecv_cov_ema', len(cov_emas), self.wcomm.rank)
            for linkname, matrices in sorted(cov_emas.items()):
                for i, matrix in enumerate(matrices):
                    data = self.wcomm.recv(self.cov_worker_rank, 0)
                    with cuda.get_device_from_array(matrix) as dev:
                        if dev.id < 0:
                            matrix[:] = data
                        else:
                            matrix[:] = chainer.cuda.to_gpu(data)
            print('sendrecv_cov_ema done', len(cov_emas), self.wcomm.rank)


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
    comm = KFACCommunicator(communicator_name='naive', debug=True)
