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

        n_group = wcomm.inter_size // npergroup
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


if __name__ == '__main__':
    comm = KFACCommunicator(MPI.COMM_WORLD, communicator_name='naive',
                            debug=True)
