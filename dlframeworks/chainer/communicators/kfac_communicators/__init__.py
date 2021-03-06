def create_communicator(
        communicator_name='flat', mpi_comm=None, dynamic=False, debug=False):

    if mpi_comm is None:
        import mpi4py.MPI
        mpi_comm = mpi4py.MPI.COMM_WORLD

    if communicator_name == 'flat':
        from dlframeworks.chainer.communicators.kfac_communicators\
            .flat_communicator import FlatCommunicator
        return FlatCommunicator(
            mpi_comm=mpi_comm, dynamic=dynamic, debug=debug)
    else:
        raise ValueError(
            'Unrecognized communicator: "{}"'.format(communicator_name))
