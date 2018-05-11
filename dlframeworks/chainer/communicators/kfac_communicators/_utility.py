import itertools

from dlframeworks.chainer.utils import create_mpi_print


def extract(fisher_blocks, indices, extractors):
    arrays = []
    for local_indices in indices:
        if len(local_indices) == 0:
            arrays.append([])
        else:
            local_arrays = []
            for index in local_indices:
                for extractor in extractors:
                    for array in extractor(fisher_blocks[index]):
                        local_arrays.append(array)
            arrays.append(local_arrays)
    return arrays


def extract_cov_emas(fisher_block):
    ret = []
    if fisher_block.cov_emas is not None:
        for cov_ema in fisher_block.cov_emas:
            ret.append(cov_ema)
    return ret


def extract_grads(fisher_block):
    ret = []
    for _, param in sorted(fisher_block.link.namedparams()):
        if param.grad is not None:
            ret.append(param.grad)
    return ret


def extract_kfgrads(fisher_block):
    ret = []
    for _, param in sorted(fisher_block.link.namedparams()):
        if hasattr(param, 'kfgrad') and param.kfgrad is not None:
            ret.append(param.kfgrad)
    return ret


def get_nelems(arrays):
    nelems = 0
    for array in list(itertools.chain(*arrays)):  # flatten arrays
        nelems += array.size
    return nelems


def get_sendcounts_and_displs(arrays):
    sendcounts = []
    displs = []
    sendcount_offset = 0
    for local_arrays in arrays:
        sendcount = 0
        for array in local_arrays:
            sendcount += array.size
        sendcounts.append(sendcount)
        displs.append(sendcount_offset)
        sendcount_offset += sendcount
    return sendcounts, displs


def pack(arrays, gpu_buf, sizeof_dtype):
    buf_offset = 0
    for array in list(itertools.chain(*arrays)):  # flatten arrays
        nbytes = array.size * sizeof_dtype
        gpu_buf.from_device(array, nbytes, buf_offset)
        buf_offset += nbytes


def unpack(arrays, gpu_buf, sizeof_dtype):
    buf_offset = 0
    for array in list(itertools.chain(*arrays)):  # flatten arrays
        nbytes = array.size * sizeof_dtype
        gpu_buf.to_device(array, nbytes, buf_offset)
        buf_offset += nbytes


def allocate_kfgrads(fisher_blocks):
    for fisher_block in fisher_blocks:
        for _, param in sorted(fisher_block.link.namedparams()):
            if param.grad is None:
                continue
            if not hasattr(param, 'kfgrad'):
                kfgrad = param.grad.copy()
                kfgrad.fill(0.)
                setattr(param, 'kfgrad', kfgrad)


def print_debug_message(mpi_comm, arrays, prefix):
    mpi_print = create_mpi_print(mpi_comm)
    idx = 0
    for array in list(itertools.chain(*arrays)):  # flatten arrays
        mpi_print('{} {} MEAN {}'.format(
            prefix, idx, array.mean()))
        idx += 1
