from dlframeworks.chainer.utils import get_link


def reduce_scatterv_pack(dictionary, divided_linknames, gpu_buf, sizeof_dtype):
    sendcounts = []
    displs = []
    buf_offset = 0
    sendcount_offset = 0
    for linknames in divided_linknames:
        sendcount = 0
        for linkname in sorted(linknames):
            arrays = dictionary[linkname]
            for array in arrays:
                sendcount += array.size
                nbytes = array.size * sizeof_dtype
                gpu_buf.from_device(array, nbytes, buf_offset)
                buf_offset += nbytes
        sendcounts.append(sendcount)
        displs.append(sendcount_offset)
        sendcount_offset += sendcount
    return sendcounts, displs


def reduce_scatterv_unpack(dictionary, linknames, gpu_buf, sizeof_dtype):
    buf_offset = 0
    for linkname in sorted(linknames):
        arrays = dictionary[linkname]
        for array in arrays:
            nbytes = array.size * sizeof_dtype
            gpu_buf.to_device(array, nbytes, buf_offset)
            buf_offset += nbytes


def allgatherv_pack(model, divided_linknames, gpu_buf, sizeof_dtype, rank):
    sendcounts = []
    displs = []
    sendcount_offset = 0
    buf_offset = 0
    for i, linknames in enumerate(divided_linknames):
        sendcount = 0
        for linkname in sorted(linknames):
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


def allgatherv_unpack(model, linknames, gpu_buf, sizeof_dtype):
    buf_offset = 0
    for linkname in linknames:
        link = get_link(model, linkname)
        for paramname, param in sorted(link.namedparams()):
            if param.kfgrad is None:
                continue
            nbytes = param.kfgrad.size * sizeof_dtype
            gpu_buf.to_device(param.kfgrad, nbytes, buf_offset)
            buf_offset += nbytes
