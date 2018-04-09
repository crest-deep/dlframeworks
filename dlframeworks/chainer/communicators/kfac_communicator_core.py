import chainer
import numpy as np


class CPUCommunicatorCore(object):
    def __init__(self, comm):
        self.comm = comm

    def reduce_inv(self, invs):
        comm = self.comm.icomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        for linkname, matrices in sorted(invs.items()):
            for matrix in matrices:
                x = chainer.cuda.to_cpu(matrix).astype(np.float32)
                x = comm.mpi_comm.reduce(x)
                if x is None:  # I'm not the master
                    continue
                with chainer.cuda.get_device_from_array(matrix) as dev:
                    if dev.id < 0:
                        matrix[:] = x
                    else:
                        matrix[:] = chainer.cuda.to_gpu(x)

    def bcast_inv(self, invs):
        comm = self.comm.gcomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        for linkname, matrices in sorted(invs.items()):
            for matrix in matrices:
                x = chainer.cuda.to_cpu(matrix).astype(np.float32)
                x = comm.mpi_comm.bcast(x)
                with chainer.cuda.get_device_from_array(matrix) as dev:
                    if dev.id < 0:
                        matrix[:] = x
                    else:
                        matrix[:] = chainer.cuda.to_gpu(x)

    def allreduce_cov(self, covs):
        comm = self.comm.ccomm
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        for matrix in covs:
            x = chainer.cuda.to_cpu(matrix).astype(np.float32)
            x = comm.mpi_comm.allreduce(x)
            x /= comm.size
            with chainer.cuda.get_device_from_array(matrix) as dev:
                if dev.id < 0:
                    matrix[:] = x
                else:
                    matrix[:] = chainer.cuda.to_gpu(x)

    def bcast_param(self, model):
        comm = self.comm.ccomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        for name, param in sorted(model.namedparams()):
            x = chainer.cuda.to_cpu(param.data).astype(np.float32)
            x = comm.mpi_comm.bcast(x)
            with chainer.cuda.get_device_from_array(param.data) as dev:
                if dev.id < 0:
                    param.data[:] = x
                else:
                    param.data[:] = chainer.cuda.to_gpu(x)


class GPUCommunicatorCore(object):
    def __init__(self, comm, debug=False):
        self.comm = comm
        self.debug = debug

    def reduce_inv(self, invs):
        comm = self.comm.icomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        invs_chain = DummyChain(invs)
        if self.debug:
            for paramname, param in sorted(invs_chain.namedparams()):
                for rank in range(comm.size):
                    if comm.rank == rank:
                        print("""\
REDUCE_INV BEFORE: RANK: {}, PARAMNAME: {}, \
MEAN: {}, MAX: {}, MIN: {}, DTYPE: {}""".format(
                            self.comm.wcomm.rank, paramname, param.grad.mean(),
                            param.grad.max(), param.grad.min(),
                            param.grad.dtype))
                    comm.mpi_comm.Barrier()
        comm.allreduce_grad(invs_chain)
        invs_chain.unpack(invs)
        for linkname, matrices in sorted(invs.items()):
            for i, matrix in enumerate(matrices):
                matrix *= self.comm.icomm_g.size
                if self.debug:
                    for rank in range(comm.size):
                        if comm.rank == rank:
                            print("""\
REDUCE_INV AFTER: RANK: {}, PARAMNAME: {}, \
MEAN: {}, MAX: {}, MIN: {}, DTYPE: {}""".format(
                                self.comm.wcomm.rank, linkname + '/' + str(i),
                                param.grad.mean(), param.grad.max(),
                                param.grad.min(), param.grad.dtype))
                        comm.mpi_comm.Barrier()

    def bcast_inv(self, invs):
        comm = self.comm.gcomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        invs_chain = DummyChain(invs)
        if self.debug:
            for paramname, param in sorted(invs_chain.namedparams()):
                for rank in range(comm.size):
                    if comm.rank == rank:
                        print("""\
BCAST_INV BEFORE: RANK: {}, PARAMNAME: {}, \
MEAN: {}, MAX: {}, MIN: {}, DTYPE: {}""".format(
                            self.comm.wcomm.rank, paramname, param.grad.mean(),
                            param.grad.max(), param.grad.min(),
                            param.grad.dtype))
                    comm.mpi_comm.Barrier()
        comm.broadcast_data(invs_chain)
        invs_chain.unpack(invs)
        if self.debug:
            for linkname, matrices in sorted(invs.items()):
                for i, matrix in enumerate(matrices):
                    for rank in range(comm.size):
                        if comm.rank == rank:
                            print("""\
REDUCE_INV AFTER: RANK: {}, PARAMNAME: {}, \
MEAN: {}, MAX: {}, MIN: {}, DTYPE: {}""".format(
                                self.comm.wcomm.rank, linkname + '/' + str(i),
                                param.grad.mean(), param.grad.max(),
                                param.grad.min(), param.grad.dtype))
                        comm.mpi_comm.Barrier()

    def allreduce_cov(self, covs):
        comm = self.comm.ccomm
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        covs_chain = DummyChain(covs)
        if self.debug:
            for cov in covs:
                for rank in range(comm.size):
                    if comm.rank == rank:
                        print("""\
ALLREDUCE_COV BEFORE: RANK: {}, MEAN: {}, MAX: {}, MIN: {}
""".format(self.comm.wcomm.rank, cov.mean(), cov.max(), cov.min(), cov.dtype))
                    comm.mpi_comm.Barrier()
        comm.allreduce_grad(covs_chain)
        covs_chain.unpack(covs)
        if self.debug:
            for cov in covs:
                for rank in range(comm.size):
                    if comm.rank == rank:
                        print("""\
ALLREDUCE_COV BEFORE: RANK: {}, MEAN: {}, MAX: {}, MIN: {}
""".format(self.comm.wcomm.rank, cov.mean(), cov.max(), cov.min(), cov.dtype))
                    comm.mpi_comm.Barrier()

    def bcast_param(self, model):
        comm = self.comm.ccomm_g
        init_comms = getattr(comm, '_init_comms')
        init_comms()
        comm.broadcast_data(model)


class DummyLink(object):
    """A dummy link that overrides `namedparams` method"""

    def __init__(self, data):
        self._params = {}
        self._params['/'] = DummyParameter(data)

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    @property
    def data(self):
        return self._params['/'].data


class DummyChain(object):
    """A dummy chain that overrides `namedparams` method"""

    def __init__(self, data):
        self._params = {}
        if isinstance(data, dict):
            self._params = _pack_dict(data)
        elif isinstance(data, list):
            self._params = _pack_list(data)
        else:
            raise ValueError('Invalid datatype: ', type(data))

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    def unpack(self, data):
        if isinstance(data, dict):
            _unpack_dict(data, self._params)
        elif isinstance(data, list):
            _unpack_list(data, self._params)
        else:
            raise ValueError('Invalid datatype: ', type(data))


class DummyParameter(object):
    """A dummy link that overrides `grad` method"""

    def __init__(self, data):
        self._data = [data]

    @property
    def data(self):
        return self._data[0]

    @data.setter
    def data(self, data):
        self._data[0] = data

    @property
    def grad(self):
        return self._data[0]

    @grad.setter
    def grad(self, data):
        self._data[0] = data


def _pack_dict(d):
    params = {}
    for linkname, matrices in sorted(d.items()):
        digits = len(str(len(matrices) - 1))
        for i, matrix in enumerate(matrices):
            key = linkname + '/{{:0{}}}'.format(digits).format(i)
            xp = chainer.cuda.get_array_module(matrix)
            params[key] = DummyParameter(matrix.astype(xp.float32))
    return params


def _unpack_dict(d, params):
    for linkname, matrices in sorted(d.items()):
        digits = len(str(len(matrices) - 1))
        for i, matrix in enumerate(matrices):
            key = linkname + '/{{:0{}}}'.format(digits).format(i)
            matrix[:] = params[key].data


def _pack_list(l):
    params = {}
    digits = len(str(len(l) - 1))
    for i, matrix in enumerate(l):
        key = '/{{:0{}}}'.format(digits).format(i)
        xp = chainer.cuda.get_array_module(matrix)
        params[key] = DummyParameter(matrix.astype(xp.float32))
    return params


def _unpack_list(l, params):
    digits = len(str(len(l) - 1))
    for i, matrix in enumerate(l):
        key = '/{{:0{}}}'.format(digits).format(i)
        matrix[:] = params[key].data
