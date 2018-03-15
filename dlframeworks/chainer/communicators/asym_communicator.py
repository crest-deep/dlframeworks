import chainermn
from mpi4py import MPI


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


class AsymCommunicator(object):

    def __init__(self, mpi_comm, communicator_name='hierarchical', ratio=0.2,
                 debug=False):
        rank = mpi_comm.rank
        size = mpi_comm.size
        if debug:
            print_mpi = _create_print_mpi(mpi_comm)
        is_masters = [False for _ in range(size)]
        for i in range(size):
            if i < size * 0.2:
                is_masters[i] = True

        # Create MPI's intra communicator
        if is_masters[rank]:
            mpi_intracomm = mpi_comm.Split(color=0, key=rank)
        else:
            mpi_intracomm = mpi_comm.Split(color=1, key=rank)

        if debug:
            print_mpi('intra: {}/{}'.format(mpi_intracomm.rank,
                                            mpi_intracomm.size))

        # Create MPI's inter communicator
        local_leader = 0
        if is_masters[rank]:
            remote_leader = next(i for i, x in enumerate(is_masters) if not x)
            mpi_intercomm_root = MPI.ROOT if \
                mpi_intracomm.rank == local_leader else MPI.PROC_NULL
        else:
            remote_leader = next(i for i, x in enumerate(is_masters) if x)
            mpi_intercomm_root = remote_leader
        mpi_intercomm = mpi_intracomm.Create_intercomm(
            local_leader, mpi_comm, remote_leader)

        if debug:
            print_mpi('inter: {}/{}'.format(mpi_intercomm.rank,
                                            mpi_intercomm.size))
            send_buf = 0 if is_masters[rank] else 16
            recv_buf = mpi_intercomm.reduce(send_buf, root=mpi_intercomm_root)
            print_mpi('recv_buf: {}'.format(recv_buf))

        actual_comm = chainermn.create_communicator(communicator_name,
                                                    mpi_comm=mpi_intracomm)

        super(AsymCommunicator, self).__setattr__(
            'mpi_intracomm', mpi_intracomm)
        super(AsymCommunicator, self).__setattr__(
            'mpi_intercomm', mpi_intercomm)
        super(AsymCommunicator, self).__setattr__(
            'mpi_intercomm_root', mpi_intercomm_root)
        super(AsymCommunicator, self).__setattr__(
            'is_master', is_masters[rank])
        super(AsymCommunicator, self).__setattr__(
            'actual_comm', actual_comm)

    def __getattr__(self, name):
        return getattr(self.actual_comm, name)

    def __setattr__(self, name, value):
        setattr(self.actual_comm, name, value)


if __name__ == '__main__':
    comm = AsymCommunicator(MPI.COMM_WORLD, debug=True)
