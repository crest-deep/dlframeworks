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

    for linkname, matrix in invs.items():
        matrix_link = DummyLink(matrix)
        comm.gcomm_g.broadcast_data(matrix_link)
        invs[linkname] = matrix_link.data


def allreduce_cov(comm, covs):
    """Allreduce covariance matrices

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        covs (list(numpy.array)): Send buffer or recv buffer of
            covariance matrices.
    """
    for i, matrix in enumerate(covs):
        matrix_link = DummyLink(matrix)
        comm.ccomm.allreduce_grad(matrix_link)
        covs[i] = matrix_link.data


def sendrecv_param(comm,optimizer):
    """
    この実装ならoptimizerではなくtarget (chain)を引数にもらってもいいのでは?
    """
    """Send ans recieve optimizer.target (grad_master -> cov_worker)

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        optimizer (KFAC): 
    """
    is_sender = comm.is_grad_master
    is_reciever = comm.is_cov_worker
    params = list(optimizer.target.namedparams())
    params = sorted(params, params.keys())

    if is_sender:
        for param in params:
            comm.wcomm.mpi_comm.send(param[0], dest=comm.cov_worker_rank)
    elif is_reciever:
        for linkname, _ in params:
            params[linkname] = comm.wcomm.mpi_comm.recv(source = comm.grad_master_rank)


def sendrecv_cov_ema(comm,cov_ema):
    """Send and recieve cov_ema_dict (cov_worker -> inv_worker)

    Args:
        comm (chainermn._base.CommunicatorBase): Wrapped ChainerMN
            communicator.
        covs (list(numpy.array)): Send buffer or recv buffer of
            covariance matrices.
    """
    is_sender = comm.is_cov_worker
    is_reciever = comm.is_inv_worker
    def _make_presudo_odict(odic):
        i = 0
        dic = dict()
        for k,v in odic.items():
            dic[i] = (k, v)
            i += 1
        return dic

    def _make_real_odict(dic):
        odic = OrderedDict()
        for _, v in sorted(dic.items()):
            odic[v[0]] = v[1]
        return odic

    if is_sender:
        s_dic = _make_presudo_odict(cov_ema)
        comm.wcomm.mpi_comm.send(s_dic, dest=comm.inv_worker_rank)
    elif is_reciever:
        r_dic = comm.wcomm.mpi_comm.recv(source=comm.cov_worker_rank)
        r_odic = _make_real_odict(r_dic)
        for k,v in r_odic.items():
            cov_ema[k] = v


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
