class DummyLink(object):
    """A dummy link that overwride `namedparams` method"""

    def __init__(self, data):
        self._params = {}
        self._params['/'] = DummyParameter(data)

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    @property
    def data(self):
        return self._params['/'].data


class DummyParameter(object):
    """A dummy link that overwride `grad` method"""

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


def allreduce_grad(comm, optimizer):
    """Allreduce gradients calculated by backprop

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        link (chainer.Link): Model that is updated.
    """
    target = optimizer.target
    if _is_changed(optimizer):
        comm.gcomm.broadcast_data(target)
        return False
    else:
        comm.gcomm.allreduce_grad(target)
        return True


def bcast_inv(comm, invs):
    """Broadcast inverse matrices

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        invs (OrderedDict(str, numpy.array)): Send buffer or recv buffer of
            inverse matrices.
    """
    root = comm.inv_worker_rank

    for linkname, matrices in invs.items():
        for matrix in matrices:
            matrix_link = DummyLink(matrix)
            comm.gcomm_g.broadcast_data(matrix_link)
            invs[linkname] = matrix_link.data


def allreduce_cov(comm, covs):
    """Allreduce covariance matrices

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        covs (OrderedDict(str, numpy.array)): Send buffer or recv buffer of
            covariance matrices.
    """
    for linkname, matrices in covs.items():
        for matrix in matrices:
            matrix_link = DummyLink(matrix)
            comm.ccomm.allreduce_grad(matrix_link)
            matrix = matrix_link.data


def _is_changed(optimizer):
    target = optimizer.target
    previous_params = optimizer.target_params
    optimizer.target_params = [(name, param.data is not None)
                               for name, param in sorted(target.namedparams())]
    if len(previous_params) != len(optimizer.target_params):
        return True
    for param1, param2 in zip(optimizer.target_params, previous_params):
        if (param1[0] != param2[0]) or (param1[1] != param2[1]):
            return True
    return False
