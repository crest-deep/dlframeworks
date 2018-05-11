import chainer.cuda
from chainermn.communicators._memory_utility import DeviceMemory
import mpi4py.MPI

from dlframeworks.chainer.communicators.kfac_communicators \
    import kfac_communicator_base
from dlframeworks.chainer.communicators.kfac_communicators \
    import _utility


class FlatCommunicator(kfac_communicator_base.KFACCommunicatorBase):

    def __init__(self, mpi_comm, debug=False):
        super(FlatCommunicator, self).__init__(mpi_comm, debug)

        # GPU buffers
        self.gpu_buffer_a = DeviceMemory()
        self.gpu_buffer_b = DeviceMemory()

        # Assume 32 bit floating point
        self.mpi_dtype = mpi4py.MPI.FLOAT
        self.sizeof_dtype = 4

    def reduce_scatterv_grad(self, fisher_blocks, root=0):
        """Reduce and Scatterv grads and covs

        1. Extract
            grads, cov_emas -> arrays
        2. Pack
            arrays -> GPU buffer A
        3. Reduce
            GPU buffer A -> GPU buffer B
        4. Scatterv
            GPU buffer B -> GPU buffer A
        5. Unpack
            GPU buffer A -> arrays

        """
        self.setup(fisher_blocks)
        cuda_stream = chainer.cuda.Stream.null

        # We extract cov_emas and grads from fisher_blocks
        extractors = [_utility.extract_cov_emas, _utility.extract_grads]
        arrays = _utility.extract(fisher_blocks, self.indices, extractors)

        # Get total number of elements
        nelems = _utility.get_nelems(arrays)
        nbytes = nelems * self.sizeof_dtype

        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Calculate sendcounts, and calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = _utility.get_sendcounts_and_displs(arrays)

        # Pack the elements in a single buffer
        _utility.pack(arrays, self.gpu_buffer_a, self.sizeof_dtype)

        # Buffers for Reduce
        sendbuf = [self.gpu_buffer_a.buffer(nbytes), self.mpi_dtype]
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), self.mpi_dtype] if \
            self.rank == root else None

        if self.debug:
            _utility.print_debug_message(self.mpi_comm, arrays,
                                         'BEFORE REDUCE_SCATTERV')

        # We must sync before communication
        cuda_stream.synchronize()
        self.mpi_comm.Reduce(sendbuf, recvbuf, root=root)

        if not self.is_inv_worker:
            return

        # Buffers for Scatterv
        nbytes_local = sendcounts[self.inv_comm.rank] * self.sizeof_dtype
        sendbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   self.mpi_dtype] if self.rank == root else None
        recvbuf = self.gpu_buffer_a.buffer(nbytes_local)

        # We must sync before communication
        cuda_stream.synchronize()
        self.inv_comm.mpi_comm.Scatterv(sendbuf, recvbuf, root=root)

        # Unpack the all elements
        _utility.unpack(arrays[self.inv_comm.rank], self.gpu_buffer_a,
                        self.sizeof_dtype)

        if self.debug:
            _utility.print_debug_message(self.mpi_comm, arrays,
                                         'AFTER REDUCE_SCATTERV')

    def allgatherv_kfgrad(self, fisher_blocks):
        """Allgatherv kfgrads

        1. Extract
            kfgrads -> arrays
        1. Pack
            arrays -> GPU buffer A
        2. Allgatherv
            GPU buffer A -> GPU buffer B
        3. Unpack
            GPU buffer B -> arrays

        """
        # Allocate memory space for recieving kfgrads
        _utility.allocate_kfgrads(fisher_blocks)

        cuda_stream = chainer.cuda.Stream.null

        # We extract kfgrads from fisher_blocks
        extractors = [_utility.extract_kfgrads]
        arrays = _utility.extract(fisher_blocks, self.indices, extractors)

        # Get total number of elements
        nelems = _utility.get_nelems(arrays)
        nbytes = nelems * self.sizeof_dtype

        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Calculate sendcounts, and calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = _utility.get_sendcounts_and_displs(arrays)

        # Pack the elements in a single buffer
        _utility.pack(arrays[self.rank], self.gpu_buffer_a, self.sizeof_dtype)

        # Buffers for Allgatherv
        nbytes_local = sendcounts[self.rank] * self.sizeof_dtype
        sendbuf = self.gpu_buffer_a.buffer(nbytes_local)
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   self.mpi_dtype]

        if self.debug:
            _utility.print_debug_message(self.mpi_comm, arrays,
                                         'BEFORE ALLGATHERV')

        # We must sync before communication
        cuda_stream.synchronize()
        self.mpi_comm.Allgatherv(sendbuf, recvbuf)

        # Unpack the all elements
        _utility.unpack(arrays, self.gpu_buffer_b, self.sizeof_dtype)

        if self.debug:
            _utility.print_debug_message(self.mpi_comm, arrays,
                                         'AFTER ALLGATHERV')
