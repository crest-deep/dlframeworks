import argparse
import chainer
import chainermn
import numpy as np
import time

from dlframeworks.chainer.communicators import kfac_communicator_core


def _create_print_mpi(comm):
    import mpi4py.MPI
    rank = comm.rank
    size = comm.size
    host = mpi4py.MPI.Get_processor_name()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def print_mpi(*args, root=None, **kwargs):
        for i in range(size):
            if i == rank:
                if root is not None:
                    if i == root:
                        print(prefix, end='')
                        print(*args, **kwargs)
                else:
                    print(prefix, end='')
                    print(*args, **kwargs)
            comm.Barrier()
    return print_mpi


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
        timeout (int): Seconds for timeout.
        join_cov (bool): Join covariance worker and inverse worker
        use_cupy (bool): Use ChainerMN's CuPy direct communication
        check_value (bool): Check the communicated values every after
            communication, this will take much more time for communication.
    """

    def __init__(self, communicator_name='hierarchical', mpi_comm=None,
                 npergroup=1, debug=False, timeout=90, n_cov_workers=1,
                 n_inv_workers=1, join_cov=False, use_cupy=False,
                 check_value=False):
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

        # Communicator for all covariance workers and gradient master in a
        # group
        color = (group_inter_rank + 1) * (is_grad_master | is_cov_worker)
        key = 0 if is_grad_master else wcomm.rank + 1
        ccomm_g = wcomm.split(color=color, key=key)

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

        self.timeout = timeout
        self.join_cov = join_cov
        self.use_cupy = use_cupy
        self.check_value = check_value

        self.wcomm = wcomm
        self.ccomm = ccomm
        self.gcomm = gcomm
        self.ccomm_g = ccomm_g
        self.icomm_g = icomm_g
        self.gcomm_g = gcomm_g
        self.comm_core = kfac_communicator_core.CPUCommunicatorCore(self)

        self.is_grad_master = is_grad_master
        self.is_grad_worker = is_grad_worker
        self.is_cov_master = is_cov_master
        self.is_cov_worker = is_cov_worker
        self.is_inv_master = is_inv_master
        self.is_inv_worker = is_inv_worker

        self.group_inter_size = group_inter_size
        self.group_inter_rank = group_inter_rank
        self.group_intra_size = group_intra_size
        self.group_intra_rank = group_intra_rank

        self.grad_master_rank = grad_master_rank
        self.grad_worker_ranks = grad_worker_ranks
        self.cov_master_rank = cov_master_rank
        self.cov_worker_ranks = cov_worker_ranks
        self.inv_master_rank = inv_master_rank
        self.inv_worker_ranks = inv_worker_ranks

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

        # Send and recieve heart beat
        if self.is_grad_master:
            _send_heartbeat(self.wcomm.mpi_comm, self.inv_master_rank,
                            tag=(100 * self.inv_master_rank + 0))
        elif self.is_inv_master:
            is_done = _recv_heartbeat(
                self.wcomm.mpi_comm, self.grad_master_rank,
                (100 * self.inv_master_rank + 0), self.timeout)
            if is_done:
                return True

        if self.is_inv_worker:
            self.comm_core.reduce_inv(invs)

        if not self.is_inv_master and not self.is_grad_worker:
            return

        self.comm_core.bcast_inv(invs)

    def allreduce_cov(self, covs):
        """Allreduce covariance matrices

        Args:
            covs (list(numpy/cupy.array)): Send buffer or recv buffer of
                covariance matrices.
        """
        if not self.is_cov_worker:
            return

        self.comm_core.allreduce_cov(covs)

    def sendrecv_param(self, optimizer):
        """Send or recieve parameters

        Sender is gradient master and reciever is covariance worker.

        Args:
            optimizer (chainer.Optimizer): KFAC optimizer.
        """
        if not self.is_grad_master and not self.is_cov_worker:
            return

        if self.is_grad_master:
            for cov_worker_rank in self.cov_worker_ranks:
                _send_heartbeat(self.wcomm.mpi_comm, cov_worker_rank,
                                tag=(100 * cov_worker_rank + 2))
        elif self.is_cov_master:
            is_done = _recv_heartbeat(
                self.wcomm.mpi_comm, self.grad_master_rank,
                (100 * self.wcomm.rank + 2), self.timeout)
            if is_done:
                return True

        self.comm_core.bcast_param(optimizer.target)

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
                        _send(self.wcomm, matrix, self.inv_worker_rank,
                              100 * inv_worker_rank + 4)
        elif self.is_inv_worker:
            for linkname, matrices in sorted(cov_emas.items()):
                for matrix in matrices:
                    _recv(self.wcomm, matrix, self.cov_worker_rank,
                          100 * self.wcomm.rank + 4)


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


def _send(comm, array, dest, tag):
    """Send array

    Args:
        comm (chainermn.communicators.CommunicatorBase): ChainerMN communicator
        array (numpy/cupy.array): Sending array
        dest (int): Destination rank
        tag (int): Tag which is used for MPI
    """
    array_cpu = chainer.cuda.to_cpu(array).astype(np.float32)
    comm.mpi_comm.Send(array_cpu, dest=dest, tag=tag)


def _recv(comm, array, source, tag):
    """Recieve array

    Args:
        comm (chainermn.communicators.CommunicatorBase): ChainerMN communicator
        array (numpy/cupy.array): Recieving array buffer
        source (int): Source rank
        tag (int): Tag which is used for MPI
    """
    array_cpu = chainer.cuda.to_cpu(array).astype(np.float32)
    comm.mpi_comm.Recv(array_cpu, source=source, tag=tag)
    with chainer.cuda.get_device_from_array(array) as dev:
        if dev.id < 0:
            array[:] = array_cpu
        else:
            array[:] = chainer.cuda.to_gpu(array_cpu)


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
            return False
    if t >= timeout:
        # Nothing came, cancel the communication
        req.Cancel()
        return True
    return False


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
