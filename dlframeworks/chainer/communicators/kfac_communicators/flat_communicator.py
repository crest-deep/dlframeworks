import chainer.cuda
from chainermn.communicators._memory_utility import DeviceMemory
import mpi4py.MPI

from dlframeworks.chainer.communicators.kfac_communicators \
    import kfac_communicator_base
from dlframeworks.chainer.communicators.kfac_communicators \
    import _memory_utility


class FlatCommunicator(kfac_communicator_base.KFACCommunicatorBase):

    def __init__(self, mpi_comm, dynamic=False, debug=False):
        super(FlatCommunicator, self).__init__(
            mpi_comm, False, dynamic, debug)

        # GPU buffers
        self.gpu_buffer_a = DeviceMemory()
        self.gpu_buffer_b = DeviceMemory()

        # Assume 32 bit floating point
        self.mpi_dtype = mpi4py.MPI.FLOAT
        self.sizeof_dtype = 4

    def reduce_scatterv(self, model, covs, root=0):
        """Reduce and Scatterv grads and covs

        1. Extract (by reference)
            model, covs -> dictionary
        2. Pack
            dictionary -> GPU buffer A
        3. Reduce
            GPU buffer A -> GPU buffer B
        4. Scatterv
            GPU buffer B -> GPU buffer A
        5. Unpack
            GPU buffer A -> dictionary

        """
        self.setup(model)
        cuda_stream = chainer.cuda.Stream.null

        dictionary = self.reduce_scatterv_extract(model, covs)
        nelems = self.reduce_scatterv_get_nelems(dictionary)
        nbytes = nelems * self.sizeof_dtype

        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Pack the elements in a single buffer, calculate sendcounts, and
        # calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = _memory_utility.reduce_scatterv_pack(
            dictionary, self.divided_linknames, self.gpu_buffer_a,
            self.sizeof_dtype)

        # Buffers for Reduce
        sendbuf = [self.gpu_buffer_a.buffer(nbytes), self.mpi_dtype]
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), self.mpi_dtype] if \
            self.rank == root else None

        if self.debug:
            self.reduce_scatterv_debug(dictionary, 'BEFORE')

        # We must sync before communication
        cuda_stream.synchronize()
        self.mpi_comm.Reduce(sendbuf, recvbuf, root=root)

        if not self.is_worker:
            return

        # Buffers for Scatterv
        nbytes_local = sendcounts[self.invcomm.rank] * self.sizeof_dtype
        sendbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   self.mpi_dtype] if self.rank == root else None
        recvbuf = self.gpu_buffer_a.buffer(nbytes_local)

        # We must sync before communication
        cuda_stream.synchronize()
        self.invcomm.mpi_comm.Scatterv(sendbuf, recvbuf, root=root)

        # Unpack the all elements
        _memory_utility.reduce_scatterv_unpack(
            dictionary, self.divided_linknames[self.invcomm.rank],
            self.gpu_buffer_a, self.sizeof_dtype)

        if self.debug:
            self.reduce_scatterv_debug(dictionary, 'AFTER')

    def allgatherv(self, model):
        """Allgatherv kfgrads

        1. Pack
            kfgrads -> GPU buffer A
        2. Allgatherv
            GPU buffer A -> GPU buffer B
        3. Unpack
            GPU buffer B -> kfgrads

        """
        cuda_stream = chainer.cuda.Stream.null

        nelems = self.allgatherv_get_nelems(model)
        nbytes = nelems * self.sizeof_dtype

        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Pack the elements in a single buffer, calculate sendcounts, and
        # calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = _memory_utility.allgatherv_pack(
            model, self.divided_linknames, self.gpu_buffer_a,
            self.sizeof_dtype, self.rank)

        # Buffers for Allgatherv
        nbytes_local = sendcounts[self.rank] * self.sizeof_dtype
        sendbuf = self.gpu_buffer_a.buffer(nbytes_local)
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   self.mpi_dtype]

        if self.debug:
            self.allgatherv_debug(model, 'BEFORE')

        # We must sync before communication
        cuda_stream.synchronize()
        self.mpi_comm.Allgatherv(sendbuf, recvbuf)

        # Unpack the all elements
        _memory_utility.allgatherv_unpack(
            model, self.linknames, self.gpu_buffer_b, self.sizeof_dtype)

        if self.debug:
            self.allgatherv_debug(model, 'AFTER')
