import argparse
import chainer.cuda
import chainermn
from chainermn.communicators import _memory_utility
import numpy

from dlframeworks.chainer.utils import create_mpi_print


class KFACCommunicator(object):
    """KFAC communicator

    Args:
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, ``pure_nccl``, or
          ``single_node``)
        mpi_comm: MPI4py communicator
    """

    def __init__(self, communicator_name='hierarchical', mpi_comm=None,
                 debug=False):
        from mpi4py import MPI
        if mpi_comm is None:
            mpi_comm = MPI.COMM_WORLD
        mpi_dtype = MPI.FLOAT  # 32 bit
        sizeof_dtype = 4  # 32 bit

        mpi_print = create_mpi_print(mpi_comm)

        # Create ChainerMN communicator for all processes
        mpi_print('Creating ChainerMN communicator...')
        comm = chainermn.create_communicator(
            communicator_name=communicator_name, mpi_comm=mpi_comm)
        mpi_print('Creating ChainerMN communicator DONE')

        self.comm = comm
        self.invcomm = None
        self.inv_assigned = False
        self.debug = debug

        self.mpi_dtype = mpi_dtype
        self.sizeof_dtype = sizeof_dtype

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def setup(self, model):
        if self.invcomm is not None:
            return

        linknames = get_linknames(model)
        rank = self.comm.rank
        size = self.comm.size
        if rank < len(linknames):
            self.inv_assigned = True

        if size <= len(linknames):
            self.invcomm = self.comm
        else:
            color = int(self.inv_assigned)
            key = rank
            self.invcomm = self.comm.split(color, key)

    def reduce_scatterv_get_nelems(self, model, covs, linknames):
        nelems = 0
        for linkname in linknames:
            if linkname in covs.keys():
                for cov in covs[linkname]:
                    nelems += cov.size
            link = get_link(model, linkname)
            for _, param in sorted(link.namedparams()):
                if param.grad is not None:
                    nelems += param.grad.size
        return nelems

    def reduce_scatterv_pack_and_get_sendconts(
            self, model, covs, divided_linknames, gpu_buf, sizeof_dtype):
        sendcounts = []
        displs = []
        buf_offset = 0
        sendcount_offset = 0
        for linknames_local in divided_linknames:
            sendcount = 0
            for linkname in sorted(linknames_local):
                if linkname in covs.keys():
                    for cov in covs[linkname]:
                        sendcount += cov.size
                        nbytes = cov.size * sizeof_dtype
                        gpu_buf.from_device(cov, nbytes, buf_offset)
                        buf_offset += nbytes
                link = get_link(model, linkname)
                for _, param in sorted(link.namedparams()):
                    if param.grad is not None:
                        sendcount += param.grad.size
                        nbytes = param.grad.size * sizeof_dtype
                        gpu_buf.from_device(param.grad, nbytes, buf_offset)
                        buf_offset += nbytes
            sendcounts.append(int(sendcount))
            displs.append(int(sendcount_offset))
            sendcount_offset += sendcount
        return sendcounts, displs

    def reduce_scatterv_unpack(self, model, covs, linknames, gpu_buf,
                               sizeof_dtype):
        buf_offset = 0
        for linkname in sorted(linknames):
            if linkname in covs.keys():
                for cov in covs[linkname]:
                    nbytes = cov.size * sizeof_dtype
                    gpu_buf.to_device(cov, nbytes, buf_offset)
                    buf_offset += nbytes
            link = get_link(model, linkname)
            for _, param in sorted(link.namedparams()):
                nbytes = param.grad.size * sizeof_dtype
                gpu_buf.to_device(param.grad, nbytes, buf_offset)
                buf_offset += nbytes

    def reduce_scatterv_debug(self, comm, model, covs, linknames, prefix):
        mpi_print = create_mpi_print(comm.mpi_comm)
        idx = 0
        for linkname in sorted(linknames):
            if linkname in covs.keys():
                for cov in covs[linkname]:
                    mpi_print("""\
{prefix} REDUCE_SCATTERV IDX {idx} COV_MEAN {mean}""".format(
                        prefix=prefix, idx=idx, mean=cov.mean()))
                    idx += 1
            link = get_link(model, linkname)
            for _, param in sorted(link.namedparams()):
                mpi_print("""\
{prefix} REDUCE_SCATTERV IDX {idx} GRAD_MEAN {mean}""".format(
                    prefix=prefix, idx=idx, mean=param.grad.mean()))
                idx += 1

    def reduce_scatterv(self, model, covs, root=0):
        """Reduce and Scatterv grads and covs

        grads, covs  ----> GPU buffer A (pack)
        GPU buffer A ----> GPU buffer B (Reduce)
        GPU buffer B ----> GPU buffer A (Scatterv)
        GPU buffer A ----> grads, covs  (unpack)
        """
        mpi_dtype = self.mpi_dtype
        sizeof_dtype = self.sizeof_dtype
        comm = self.comm
        invcomm = self.invcomm
        cuda_stream = chainer.cuda.Stream.null

        # Get the linknames and divide it
        linknames = get_linknames(model)
        divided_linknames = divide_linknames(linknames, invcomm.size)
        print(len(linknames))
        print(len(divided_linknames))

        # Calculate the total bytes of elements in grads and covs
        nelems = self.reduce_scatterv_get_nelems(model, covs, linknames)
        nbytes = nelems * sizeof_dtype

        # Allocate memory if not
        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Pack the elements in a single buffer, calculate sendcounts, and
        # calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = self.reduce_scatterv_pack_and_get_sendconts(
            model, covs, divided_linknames, self.gpu_buffer_a, sizeof_dtype)

        # Buffers for Reduce
        sendbuf = [self.gpu_buffer_a.buffer(nbytes), mpi_dtype]
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), mpi_dtype] if \
            comm.rank == root else None

        # Print debug messages (invokes MPI_Barrier() many times inside)
        if self.debug:
            self.reduce_scatterv_debug(comm, model, covs, linknames, 'BEFORE')

        # We must sync before communication
        cuda_stream.synchronize()
        comm.mpi_comm.Reduce(sendbuf, recvbuf, root=root)

        if not self.inv_assigned:
            return

        # Buffers for Scatterv
        sendbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   mpi_dtype] if comm.rank == root else None
        recvbuf = self.gpu_buffer_a.buffer(
            sendcounts[invcomm.rank] * sizeof_dtype)

        # We must sync before communication
        cuda_stream.synchronize()
        invcomm.mpi_comm.Scatterv(sendbuf, recvbuf, root=root)

        # Unpack the all elements
        self.reduce_scatterv_unpack(
            model, covs, divided_linknames[invcomm.rank], self.gpu_buffer_a,
            sizeof_dtype)

        # Print debug messages (invokes MPI_Barrier() many times inside)
        if self.debug:
            self.reduce_scatterv_debug(invcomm, model, covs, linknames, 'AFTER')

    def allgatherv_get_nelems(self, model, linknames):
        nelems = 0
        for linkname in linknames:
            link = get_link(model, linkname)
            for _, param in sorted(link.namedparams()):
                if param.kfgrad is None:
                    continue
                nelems += param.kfgrad.size
        return nelems

    def allgatherv_pack_and_get_sendcounts(
            self, model, divided_linknames, gpu_buf, sizeof_dtype, rank):
        sendcounts = []
        displs = []
        sendcount_offset = 0
        buf_offset = 0
        for i, linknames_local in enumerate(divided_linknames):
            sendcount = 0
            for linkname in linknames_local:
                link = get_link(model, linkname)
                for paramname, param in sorted(link.namedparams()):
                    if param.kfgrad is None:
                        continue
                    sendcount += param.kfgrad.size
                    if i == rank:
                        nbytes = param.kfgrad.size * sizeof_dtype
                        gpu_buf.from_device(param.kfgrad, nbytes, buf_offset)
                        buf_offset += nbytes
            sendcounts.append(sendcount)
            displs.append(sendcount_offset)
            sendcount_offset += sendcount
        return sendcounts, displs

    def allgatherv_unpack(self, model, linknames, gpu_buf, sizeof_dtype):
        buf_offset = 0
        for linkname in linknames:
            link = get_link(model, linkname)
            for paramname, param in sorted(link.namedparams()):
                if param.kfgrad is None:
                    continue
                nbytes = param.kfgrad.size * sizeof_dtype
                gpu_buf.to_device(param.kfgrad, nbytes, buf_offset)
                buf_offset += nbytes

    def allgatherv_debug(self, comm, model, prefix):
        mpi_print = create_mpi_print(comm.mpi_comm)
        idx = 0
        for _, param in sorted(model.namedparams()):
            mpi_print("""\
{prefix} ALLGATHERV IDX {idx} KFGRAD_MEAN {mean}""".format(
                prefix=prefix, idx=idx, mean=param.kfgrad.mean()))
            idx += 1

    def allgatherv(self, model):
        """Step3: Allgatherv kfgrads

        kfgrads      ----> GPU buffer A (pack)
        GPU buffer A ----> GPU buffer B (Allgatherv)
        GPU buffer B ----> kfgrads      (unpack)
        """
        mpi_dtype = self.mpi_dtype
        sizeof_dtype = self.sizeof_dtype
        comm = self.comm
        invcomm = self.invcomm
        cuda_stream = chainer.cuda.Stream.null

        # Get the linknames and divide it
        linknames = get_linknames(model)
        divided_linknames = divide_linknames(linknames, invcomm.size)

        # Calculate the total number of elements in kfgrads
        nelems = self.allgatherv_get_nelems(model, linknames)
        nbytes = nelems * sizeof_dtype

        # Allocate memory if not
        self.gpu_buffer_a.assign(nbytes)
        self.gpu_buffer_b.assign(nbytes)

        # Pack the elements in a single buffer, calculate sendcounts, and
        # calculate displs
        # - sendcounts: the number of elements to send to each process
        # - displs: the displacements where each segment begins
        sendcounts, displs = self.allgatherv_pack_and_get_sendcounts(
            model, divided_linknames, self.gpu_buffer_a, sizeof_dtype,
            comm.rank)
        mpi_print = create_mpi_print(comm.mpi_comm)
        mpi_print(displs, root=0)
        mpi_print(sendcounts, root=0)

        # Buffers for Allgatherv
        sendbuf = self.gpu_buffer_a.buffer(
            sendcounts[comm.rank] * sizeof_dtype)
        recvbuf = [self.gpu_buffer_b.buffer(nbytes), sendcounts, displs,
                   mpi_dtype]

        if self.debug:
            self.allgatherv_debug(comm, model, 'BEFORE')

        # We must sync before communication
        cuda_stream.synchronize()
        comm.mpi_comm.Allgatherv(sendbuf, recvbuf)

        # Unpack the all elements
        self.allgatherv_unpack(model, linknames, self.gpu_buffer_b,
                               sizeof_dtype)

        if self.debug:
            self.allgatherv_debug(comm, model, 'AFTER')


def get_link(model, name):
    for linkname, link in model.namedlinks():
        if linkname == name:
            return link


def get_linknames(model):
    linknames = set()
    for paramname, param in model.namedparams():
        linkname = paramname[:paramname.rfind('/')]
        linknames.add(linkname)
    linknames = list(linknames)
    linknames = sorted(linknames)
    return linknames


def divide_linknames(linknames, num):
    divided_linknames = numpy.array(numpy.array_split(linknames, num))
    return divided_linknames.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    comm = KFACCommunicator(communicator_name='hierarchical', debug=True)
