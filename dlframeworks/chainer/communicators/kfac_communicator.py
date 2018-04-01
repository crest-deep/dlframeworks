import argparse
import chainer
import chainermn
import numpy as np
import time


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
        timeout (int): Minutes for timeout.
    """

    def __init__(self, communicator_name='hierarchical', mpi_comm=None,
                 npergroup=1, debug=False, timeout=90, n_cov_workers=1,
                 n_inv_workers=1, join_cov=False):
        if mpi_comm is None:
            import mpi4py.MPI
            mpi_comm = mpi4py.MPI.COMM_WORLD

        if debug:
            print_mpi = _create_print_mpi(mpi_comm)

        # World communicator which all processes contains
        wcomm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)

        if not isinstance(n_cov_workers, int) or n_cov_workers < 1:
            raise ValueError('Number of cov_worker must be positive int')
        if not isinstance(n_inv_workers, int) or n_inv_workers < 1:
            raise ValueError('Number of inv_worker must be positive int')
        if not isinstance(npergroup, int) or npergroup < 1:
            raise ValueError('Number of nodes per group must positive int')

        n_groups = wcomm.inter_size // npergroup
        if n_groups == 0:
            n_groups = 1

        if wcomm.size < n_groups * (n_cov_workers + n_inv_workers + 1):
            raise ValueError('Number of processes is not sufficient')

        group_lst = np.array_split(np.arange(wcomm.inter_size), n_groups)

        group_inter_size = n_groups  # Number of groups
        group_inter_rank = 0         # Group ID
        group_intra_size = 0         # Number of processes in this group
        group_intra_rank = 0         # Process ID in this group

        for i, group in enumerate(group_lst):
            if wcomm.inter_rank in group:
                group_inter_rank = i
                group_intra_size = len(group) * wcomm.intra_size
                j = np.where(group == wcomm.inter_rank)[0][0]
                k = wcomm.intra_rank
                group_intra_rank = j * wcomm.intra_size + k

        max_workers = n_cov_workers + 1 if join_cov else \
            n_cov_workers + n_inv_workers + 1
        if group_intra_size < max_workers:
            raise ValueError('Number of processes is not sufficient')

        is_grad_master = 0
        is_grad_worker = 0
        is_cov_master = 0
        is_cov_worker = 0
        is_inv_worker = 0
        is_inv_master = 0

        head = 0
        if group_intra_rank == head:
            is_cov_master = 1
        for i in range(head, n_cov_workers + head):
            if group_intra_rank == i:
                is_cov_worker = 1
            head += 1
        if group_intra_rank == head:
            is_inv_master = 1
        for i in range(head, n_inv_workers + head):
            if group_intra_rank == i:
                is_inv_worker = 1
            head += 1
        if not (is_cov_worker | is_inv_worker):
            if group_intra_rank == head:
                is_grad_master = 1
            is_grad_worker = 1

        if join_cov:
            if is_cov_worker:
                is_inv_worker = 1
                if is_cov_master:
                    is_inv_master = 1
            elif is_inv_worker:
                is_inv_worker = 0
                is_grad_worker = 1
                if is_inv_master:
                    is_inv_master = 0

        # Communicator for all gradient workers
        gcomm = wcomm.split(color=is_grad_worker, key=wcomm.rank)

        # Communicator for all covariance workers
        ccomm = wcomm.split(color=is_cov_worker, key=wcomm.rank)

        # Communicator for all inverse workers in a group
        color = (group_inter_rank + 1) * is_inv_worker
        key = 0 if is_inv_master else wcomm.rank + 1
        icomm_g = wcomm.split(color=color, key=key)

        # Communicator for inverse master and all gradient workers in a group
        color = (group_inter_rank + 1) * (is_grad_worker | is_inv_master)
        key = 0 if is_inv_master else wcomm.rank + 1
        gcomm_g = wcomm.split(color=color, key=key)

        send_buf = [
            is_grad_master,
            is_grad_worker,
            is_cov_master,
            is_cov_worker,
            is_inv_master,
            is_inv_worker,
            group_inter_rank,
            group_intra_rank,
            group_inter_size,
            group_intra_size,
            wcomm.rank,
        ]
        # ======== COMMUNICATION ========
        # get all flags from all processes
        recv_buf = wcomm.mpi_comm.allgather(send_buf)
        # ===============================

        grad_worker_ranks = []
        cov_worker_ranks = []
        inv_worker_ranks = []
        for flags in recv_buf:
            if group_inter_rank == flags[6]:
                if flags[1] == 1:
                    grad_worker_ranks.append(flags[-1])
                    if flags[0] == 1:
                        grad_master_rank = flags[-1]
                if flags[3] == 1:
                    cov_worker_ranks.append(flags[-1])
                    if flags[2] == 1:
                        cov_master_rank = flags[-1]
                if flags[5] == 1:
                    inv_worker_ranks.append(flags[-1])
                    if flags[4] == 1:
                        inv_master_rank = flags[-1]

        if debug:
            print_mpi("""---->
Comm:         {} / {}
Group inter:  {} / {}
Group intra:  {} / {}
Flag:         {}
Grad master:  {}
Grad workers: {}
Cov master:   {}
Cov workers:  {}
Inv master:   {}
Inv workers:  {}
--------------------------------""".format(
                wcomm.rank, wcomm.size,
                group_inter_rank, group_inter_size,
                group_intra_rank, group_intra_size,
                [is_grad_master, is_grad_worker,
                 is_cov_master, is_cov_worker,
                 is_inv_master, is_inv_worker],
                grad_master_rank,
                grad_worker_ranks,
                cov_master_rank,
                cov_worker_ranks,
                inv_master_rank,
                inv_worker_ranks))

        super(KFACCommunicator, self).__setattr__(
            'timeout', timeout)
        super(KFACCommunicator, self).__setattr__(
            'join_cov', join_cov)
        super(KFACCommunicator, self).__setattr__(
            'wcomm', wcomm)
        super(KFACCommunicator, self).__setattr__(
            'ccomm', ccomm)
        super(KFACCommunicator, self).__setattr__(
            'gcomm', gcomm)
        super(KFACCommunicator, self).__setattr__(
            'icomm_g', icomm_g)
        super(KFACCommunicator, self).__setattr__(
            'gcomm_g', gcomm_g)
        super(KFACCommunicator, self).__setattr__(
            'is_grad_master', is_grad_master)
        super(KFACCommunicator, self).__setattr__(
            'is_grad_worker', is_grad_worker)
        super(KFACCommunicator, self).__setattr__(
            'is_cov_master', is_cov_master)
        super(KFACCommunicator, self).__setattr__(
            'is_cov_worker', is_cov_worker)
        super(KFACCommunicator, self).__setattr__(
            'is_inv_master', is_inv_master)
        super(KFACCommunicator, self).__setattr__(
            'is_inv_worker', is_inv_worker)
        super(KFACCommunicator, self).__setattr__(
            'group_inter_size', group_inter_size)
        super(KFACCommunicator, self).__setattr__(
            'group_inter_rank', group_inter_rank)
        super(KFACCommunicator, self).__setattr__(
            'group_intra_size', group_intra_size)
        super(KFACCommunicator, self).__setattr__(
            'group_intra_rank', group_intra_rank)
        super(KFACCommunicator, self).__setattr__(
            'grad_master_rank', grad_master_rank)
        super(KFACCommunicator, self).__setattr__(
            'grad_worker_ranks', grad_worker_ranks)
        super(KFACCommunicator, self).__setattr__(
            'cov_master_rank', cov_master_rank)
        super(KFACCommunicator, self).__setattr__(
            'cov_worker_ranks', cov_worker_ranks)
        super(KFACCommunicator, self).__setattr__(
            'inv_master_rank', inv_master_rank)
        super(KFACCommunicator, self).__setattr__(
            'inv_worker_ranks', inv_worker_ranks)

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

        Inverse master sends A^-1 and G^-1 to all gradient workers.

        Args:
            invs (OrderedDict(str, list(numpy/cupy.array))): Send buffer or
                recieve buffer of inverse matrices.
        """
        if not self.is_grad_worker and not self.is_inv_worker: 
            return

        if self.is_grad_master:
            _send_heartbeat(self.wcomm.mpi_comm, self.inv_master_rank,
                            tag=(100 * self.inv_master_rank + 0))
        elif self.is_inv_master:
            _recv_heartbeat(self.wcomm.mpi_comm, self.grad_master_rank,
                            (100 * self.inv_master_rank + 0), self.timeout)

        # Reduce inverse (Allreduce)
        # Assume that all inverse worker have memory space with value 0
        if self.is_inv_worker:
            for linkname, matrices in sorted(invs.items()):
                for i, matrix in enumerate(matrices):
                    matrix_link = DummyLink(matrix)
                    self.icomm_g.allreduce_grad(matrix_link)
                    matrix[:] = matrix_link.data * self.icomm_g.size

        if not self.is_inv_master and not self.is_grad_worker:
            return

        # Broadcast inverse
        for linkname, matrices in sorted(invs.items()):
            for i, matrix in enumerate(matrices):
                matrix_link = DummyLink(matrix)
                # Broadcast performs on either GPU or CPU
                self.gcomm_g.broadcast_data(matrix_link)
                matrix[:] = matrix_link.data

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
            matrix[:] = matrix_link.data

    def sendrecv_param(self, optimizer):
        """Send or recieve parameters

        Sender is gradient master and reciever is covariance worker.

        Args:
            optimizer (chainer.Optimizer): KFAC optimizer.
        """
        is_sender = self.is_grad_master
        is_reciever = self.is_cov_worker

        if is_sender:
            for cov_worker_rank in self.cov_worker_ranks:
                _send_heartbeat(self.wcomm.mpi_comm, cov_worker_rank,
                                tag=(100 * cov_worker_rank + 2))
                # send parameter
                for name, param in sorted(optimizer.target.namedparams()):
                    data = param.data
                    data = chainer.cuda.to_cpu(data).astype(np.float32)
                    self.wcomm.send(data, cov_worker_rank, 0)
        elif is_reciever:
            _recv_heartbeat(self.wcomm.mpi_comm, self.grad_master_rank,
                            (100 * self.wcomm.rank + 2), self.timeout)
            # recieve parameter
            for name, param in sorted(optimizer.target.namedparams()):
                data = self.wcomm.recv(self.grad_master_rank, 0)
                with chainer.cuda.get_device_from_array(param.data) as dev:
                    if dev.id < 0:
                        param.data[:] = data
                    else:
                        param.data[:] = chainer.cuda.to_gpu(data)

    def sendrecv_cov_ema(self, cov_emas):
        """Send or recieve covariance EMAs

        Covariance workers send cov_emas to inverse workers.

        Args:
            cov_emas (dict(str, list(numpy/cupy.array))): Send buffer or
                recieve buffer of covariance EMAs.
        """
        if not self.is_cov_master and not self.is_inv_worker:
            return
        if self.join_cov:
            return

        # Assuming cov_master has all cov_emas
        if self.is_cov_master:
            for inv_worker_rank in self.inv_worker_ranks:
                for linkname, matrices in sorted(cov_emas.items()):
                    for matrix in matrices:
                        matrix = chainer.cuda.to_cpu(matrix).astype(np.float32)
                        self.wcomm.send(matrix, inv_worker_rank,
                                        (100 * inv_worker_rank + 3))
        elif self.is_inv_worker:
            for linkname, matrices in sorted(cov_emas.items()):
                for matrix in matrices:
                    data = self.wcomm.recv(self.cov_master_rank,
                                           (100 * self.wcomm.rank + 3))
                    with chainer.cuda.get_device_from_array(matrix) as dev:
                        if dev.id < 0:
                            matrix[:] = data
                        else:
                            matrix[:] = chainer.cuda.to_gpu(data)


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


def _send_heartbeat(comm, dest, tag):
    # send heartbeat
    beat = 1
    comm.send(beat, dest=dest, tag=tag)
    return True


def _recv_heartbeat(comm, source, tag, timeout):
    # recieve heartbeat
    req = comm.irecv(source=source, tag=tag)
    t = 0
    while t < timeout:
        flag, status = req.test()
        if not flag:
            time.sleep(10)
            t += 10
        else:
            return True
    if t >= timeout:
        # Nothing came, cancel the communication
        req.Cancel()
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cov_workers', type=int, default=1)
    parser.add_argument('--n_inv_workers', type=int, default=1)
    parser.add_argument('--join-cov', action='store_true', default=False)
    args = parser.parse_args()
    comm = KFACCommunicator(communicator_name='naive', debug=True,
                            n_cov_workers=args.n_cov_workers,
                            n_inv_workers=args.n_inv_workers,
                            join_cov=args.join_cov)
