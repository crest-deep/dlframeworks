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


def _split(arr, n):
    N = len(arr)
    for i in range(n):
        head = (i * N) // n
        tail = ((i + 1) * N) // n
        yield arr[head:tail]


class AsymCommunicator(object):

    def __init__(self, mpi_comm, communicator_name='hierarchical', ratio=0.2,
                 debug=False):
        if debug:
            print_mpi = _create_print_mpi(mpi_comm)

        worldcomm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)

        # Create worker group
        if worldcomm.size * ratio >= worldcomm.inter_size:
            n_masters = worldcomm.inter_size
        else:
            n_masters = max(int(worldcomm.size * ratio), 1)
        nodes = list(range(worldcomm.inter_size))
        groups = list(_split(nodes, n_masters))
        is_master = 0
        for i, group in enumerate(groups):
            if worldcomm.inter_rank in group:
                my_group = i
            if worldcomm.inter_rank == group[0] and worldcomm.intra_rank == 0:
                is_master = 1

        # Create MPI intra communicator for allreducing gradients
        intra_comm_a = worldcomm.split(color=is_master, key=worldcomm.rank)

        if debug:
            print_mpi('intra_a: {}/{}'.format(intra_comm_a.rank,
                                              intra_comm_a.size))

        # Create MPI intra communicator for broadcasting/collecting activations
        # and gradients inverse matrices
        intra_comm_b = worldcomm.split(color=my_group, key=worldcomm.rank)

        if debug:
            print_mpi('intra_b: {}/{}'.format(intra_comm_b.rank,
                                              intra_comm_b.size))

        super(AsymCommunicator, self).__setattr__(
            'worldcomm', worldcomm)
        super(AsymCommunicator, self).__setattr__(
            'intra_comm_a', intra_comm_a)
        super(AsymCommunicator, self).__setattr__(
            'intra_comm_b', intra_comm_b)
        super(AsymCommunicator, self).__setattr__(
            'my_group', my_group)
        super(AsymCommunicator, self).__setattr__(
            'is_master', is_master)

    def __getattr__(self, name):
        return getattr(self.intra_comm_a, name)

    def __setattr__(self, name, value):
        setattr(self.intra_comm_a, name, value)


if __name__ == '__main__':
    comm = AsymCommunicator(MPI.COMM_WORLD, communicator_name='naive', debug=True)
