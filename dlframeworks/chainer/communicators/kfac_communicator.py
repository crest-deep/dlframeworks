import argparse
import chainermn
from chainermn.communicators import _memory_utility

from dlframeworks.chainer.utils import create_mpi_print


class KFACCommunicator(object):
    """KFAC communicator

    Args:
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, ``pure_nccl``, or
          ``single_node``)
        mpi_comm: MPI4py communicator
    """

    def __init__(self, communicator_name='hierarchical', mpi_comm=None):
        from mpi4py import MPI
        if mpi_comm is None:
            mpi_comm = MPI.COMM_WORLD
        mpi_dtype = MPI.FLOAT  # 32 bit
        sizeof_dtype = 4  # 32 bit

        print_mpi = create_mpi_print(mpi_comm)

        # Create ChainerMN communicator for all processes
        print_mpi('Creating ChainerMN communicator...')
        comm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)
        print_mpi('Creating ChainerMN communicator DONE')

        self.comm = comm
        self.invcomm = None
        self.is_inv_assigned = False

        self.mpi_dtype = mpi_dtype
        self.sizeof_dtype = sizeof_dtype
        self.print_mpi = print_mpi

        self.n_elems_total_in_reduce_scatterv = -1
        self.n_elems_total_in_allgatherv = -1
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def init_invcomm(self, covs):
        if self.init_invcomm is not None:
            return
        if self.comm.size <= len(covs):
            self.is_inv_assigned = True
            self.init_invcomm = self.comm
        else:
            if self.comm.rank < len(covs):
                self.is_inv_assigned = True
            else:
                self.is_inv_assigned = False


    def reduce_scatterv(self, model, covs, root=0):
        """Reduce and Scatterv grads and covs

        grads, covs  ----> GPU buffer A (pack)
        GPU buffer A ----> GPU buffer B (Reduce)
        GPU buffer B ----> GPU buffer A (Scatterv)
        GPU buffer A ----> grads, covs  (unpack)
        """
        mpi_dtype = self.mpi_dtype
        sizeof_dtype = self.sizeof_dtype
        stream = None  # use null stream
        comm = self.comm
        invcomm = self.invcomm

        # Calculate the total number of elements in grads and covs
        n_elems_total = 0
        for linkname, link in sorted(model.namedlinks()):
            for cov in covs[linkname]:
                n_elems_total += cov.size
            for _, param in sorted(link.namedparams()):
                n_elems_total += param.grad.size

        # Allocate memory if not
        self.gpu_buffer_a.assign(n_elems_total * sizeof_dtype)
        self.gpu_buffer_b.assign(n_elems_total * sizeof_dtype)

        # Pack the all elements in a single buffer
        buf_offset = 0
        for linkname, link in sorted(model.namedlinks()):
            for cov in covs[linkname]:
                size = cov.size * sizeof_dtype
                self.gpu_buffer_a.from_device(
                    cov, size, buf_offset, stream=stream)
                buf_offset += size
            for _, param in sorted(link.namedparams()):
                size = param.grad.size * sizeof_dtype
                self.gpu_buffer_a.from_device(
                    param.grad, size, buf_offset, stream=stream)
                buf_offset += size

        # Buffers for Reduce
        sendbuf = [self.gpu_buffer_a.ptr(), mpi_dtype]
        recvbuf = self.gpu_buffer_b.ptr() if comm.rank == root else None

        comm.Reduce(sendbuf, recvbuf, root=root)

        if not self.is_inv_assigned:
            return

        # Calculate the sendcounts, the number of elements to send to each
        # process, and calculate the displs, the displacements where each
        # segment begins
        sendcounts = []
        displs = []
        sendcount_offset = 0
        for linknames_local in self.linknames:
            sendcount = 0
            for linkname in sorted(linknames_local):
                link = getlink(model, linkname)
                for cov in covs[linkname]:
                    sendcount += cov.size
                for _, param in sorted(link.namedparams()):
                    sendcount += param.grad.size
            sendcounts.append(sendcount)
            displs.append(sendcount_offset)
            sendcount_offset += sendcount

        # Buffers for Scatterv
        sendbuf = [self.gpu_buffer_b.ptr(), sendcounts, displs, mpi_dtype] if \
            comm.rank == root else None
        recvbuf = self.gpu_buffer_a.ptr()

        comm.Scatterv(sendbuf, recvbuf, root=root)

        # Unpack the all elements
        buf_offset = 0
        for linkname in sorted(self.linknames[comm.rank]):
            for cov in covs[linkname]:
                size = cov.size * sizeof_dtype
                self.gpu_buffer_a.to_device(
                    cov, size, buf_offset, stream=stream)
                buf_offset += size
            for _, param in sorted(link.namedparams()):
                size = param.grad.size * sizeof_dtype
                self.gpu_buffer_a.to_device(
                    param.grad, size, buf_offset, stream=stream)
                buf_offset += size

    def allgatherv(self, model, covs):
        """Step3: Allgatherv kfgrads

        kfgrads      ----> GPU buffer A (pack)
        GPU buffer A ----> GPU buffer B (Allgatherv)
        GPU buffer B ----> kfgrads      (unpack)
        """
        mpi_dtype = self.mpi_dtype
        sizeof_dtype = self.sizeof_dtype
        stream = None  # use null stream
        comm = self.comm

        # Calculate the total number of elements in grads (same as kfgrads)
        n_elems_total = 0
        for _, param in sorted(model.namedparams()):
            n_elems_total += param.grad.size

        # Allocate memory if not
        self.gpu_buffer_a.assign(n_elems_total * sizeof_dtype)
        self.gpu_buffer_b.assign(n_elems_total * sizeof_dtype)

        # Calculate the sendcounts, the displs, and pack all local elements in
        # a single buffer
        sendcounts = []
        displs = []
        sendcount_offset = 0
        buf_offset = 0
        for i in range(comm.size):
            sendcount = 0
            if i < len(self.linknames):
                linknames_local = self.linknames[i]
                for linkname in sorted(linknames_local):
                    link = getlink(model, linkname)
                    for _, param in sorted(link.namedparams()):
                        sendcount += param.kfgrad.size
                        size = param.kfgrad.size * sizeof_dtype
                        self.gpu_buffer_a.from_device(
                            param.kfgrad, size, buf_offset, stream=stream)
                        buf_offset += size
            sendcounts.append(sendcount)
            displs.append(sendcount_offset)
            sendcount_offset += sendcount

        # Buffers for Allgatherv
        sendbuf = [self.gpu_buffer_a.ptr(), sendcounts, displs, mpi_dtype]
        recvbuf = self.gpu_buffer_b.ptr()

        comm.Allgatherv(sendbuf, recvbuf)

        # Unpack the all elements
        buf_offset = 0
        for _, param in sorted(model.namedparams()):
            size = param.grad.size * sizeof_dtype
            self.gpu_buffer_b.to_device(param.kfgrad, size, buf_offset,
                                        stream=stream)
            buf_offset += size

def getlink(link, name):
    for _name, _link in link.namedlinks():
        if _name == name:
            return _link


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    comm = KFACCommunicator(communicator_name='hierarchical', debug=True)
