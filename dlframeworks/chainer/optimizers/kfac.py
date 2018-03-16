import chainer
from chainer import optimizer
from chainer.backends import cuda
from collections import OrderedDict
import numpy as np
import time

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 20
_default_hyperparam.damping = 0.001


def _check_type(data):
    if (data is not None and
            not isinstance(data, chainer.get_array_types())):
        msg = '''numpy.ndarray or cuda.ndarray are expected.
Actual: {0}'''.format(type(data))
        raise TypeError(msg)


def cov_linear(a, g):
    # TODO CPU/GPU impl
    N, _ = a.shape
    ones = np.ones(N)
    a_plus = np.column_stack((a, ones))
    A = a_plus.T.dot(a_plus) / N
    G = g.T.dot(g) / N
    return A, G


def cov_conv2d(a, g, param_shape):
    # TODO CPU/GPU impl
    N, J, H, W = a.shape
    I, J, H_k, W_k = param_shape
    T = H * W      # number of spatial location in an input feature map
    D = H_k * W_k  # number of spatial location in a kernel
    ones = np.ones(N*T)
    a_expand = np.zeros((N*T, J*D))
    for n in range(N):
        for j in range(J):
            for h in range(H):
                for w in range(W):
                    for h_k in range(H_k):
                        for w_k in range(W_k):
                            t = h*W + w
                            d = h_k*W_k + w_k
                            h_ = h+h_k-int(H_k/2)
                            w_ = w+w_k-int(W_k/2)
                            if h_ in range(H) and w_ in range(W):
                                print('n{0} j{1} h_{2} w_{3}'.format(n, j, h_, w_))
                                a_expand[t*N + n][j*D + d] = a[n][j][h_][w_]
    a_expand_plus = np.column_stack((a_expand, ones))
    A = a_expand_plus.T.dot(a_expand_plus) / N

    N, I, H_, W_ = g.shape
    T_ = H_ * W_  # number of spatial location in an output feature map
    g_expand = np.zeros((N*T_, I))
    for n in range(N):
        for i in range(I):
            for h in range(H_):
                for w in range(W_):
                    t = h*W_ + w
                    g_expand[t*N + n][i] = g[n][i][h][w]
    G = g_expand.T.dot(g_expand) / N / T_
    return A, G


class DummyLink(object):
    """A dummy link that only has ``namedparams()`` function.

    Since ChainerMN uses origianal communicators and those communicators only
    defines ``send``, ``recv``, ``alltoall``, ``broadcast_data``, and
    ``allreduce_grad``, we cannot use common ``MPI_Allreduce`` to reduce the
    data (unless we are using ChainerMN communicator). To address this problem,
    we will define a wrapper object that behaves like a ``chainer.Link``.

    In the communication functinos of ChainerMN communicators (e.g.
    ``hierarchical_communicator``), they invoke ``extrac_params(model)`` to get
    the list of ``chainer.Parameter`` objects (``model`` is passed from
    ``allreduce_grad(model)``, defined in
    ``chainermn/communicators/_memory_utility.py``). There is no other access
    to the ``model`` argument, and it means we can replace this argument to a
    non-``chainer.Link`` object if we can handle all attribution access to the
    ``model``.

    """

    def __init__(self, data):
        self._params = {}
        self._params['/'] = DummyVariable(data)

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    def unpack(self):
        return self._params['/'].data


class DummyVariable(object):
    """A dummy variable that returns ``data`` at ``grad``.

    Similar to ``DummyLink``, this class is a wrapper class the behaves like
    ``chainer.Variable``.

    """

    def __init__(self, data):
        _check_type(data)
        self._data = [data]

    @property
    def data(self):
        return self._data[0]

    @property
    def grad(self):
        return self._data[0]

    @grad.setter
    def grad(self, data):
        _check_type(data)
        self._data[0] = data


def _kfac_backward(link, backward_main):
    """Backward function for KFAC optimizer.

    This function is invoked from ``KFAC.update()`` to:
        1. calculate backprop
        2. obtain a (activations)
        3. obtain g (gradients of activation's input).

    """
    with chainer.using_config('enable_backprop', False):
        # To obtain grads, we need to edit a file ``variable.py``
        grads = backward_main(retain_grad=True, loss_scale=None)

    namedparams = list(link.namedparams())

    def get_linkname(param):
        # Get a linkname from a parameter.
        for _name, _param in namedparams:
            if param is _param:
                # Only return linkname NOT paramname.
                return _name[:_name.rfind('/')]
        return None

    acts_dict = {}
    grads_dict = {}
    ranks_dict = {}
    for node, grads in grads.items():
        creator_node = node.creator_node  # parent function node
        if creator_node is not None:  # ignore leaf node
            if isinstance(creator_node, chainer.functions.connection.linear.LinearFunction) \
              or isinstance(creator_node, chainer.functions.connection.convolution_2d.Convolution2DFunction):
                (acts, param) = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                assert linkname is not None, 'linkname cannot be None.'
                acts_dict[linkname] = acts.data    # numpy or cupy .ndarray
                grads_dict[linkname] = grads.data  # numpy or cupy .ndarray
                ranks_dict[linkname] = creator_node.rank
    # Must be sorted by its key to communicate between processes
    acts_dict = OrderedDict(sorted(acts_dict.items(), key=lambda t: t[0]))
    grads_dict = OrderedDict(sorted(grads_dict.items(), key=lambda t: t[0]))
    ranks_dict = OrderedDict(sorted(ranks_dict.items(), key=lambda t: t[0]))
    return acts_dict, grads_dict, ranks_dict


class KFACUpdateRule(chainer.optimizer.UpdateRule):

    def __init__(self, parent_hyperparam=None):
        super(KFACUpdateRule, self).__init__(
            parent_hyperparam or _default_hyperparam)

    def update_core_cpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        param.data -= self.hyperparam.lr * grad

    def update_core_gpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'ngd')(grad, self.hyperparam.lr, param.data)


class KFAC(chainer.optimizer.GradientMethod):

    def __init__(self, communicator=None, inv_server=False,
                 lr=_default_hyperparam.lr,
                 cov_ema_decay=_default_hyperparam.cov_ema_decay,
                 inv_freq=_default_hyperparam.inv_freq,
                 damping=_default_hyperparam.damping):
        super(KFAC, self).__init__()
        self.communicator = communicator
        self.inv_server = inv_server
        self.hyperparam.lr = lr
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.target_params = []
        self.acts_dict = None
        self.grads_dict = None
        self.ranks_dict = None
        self.cov_ema_dict = {}
        self.inv_dict = {}

        self._require_communication = False
        if communicator is not None:
            if communicator.size > 1:
                self._require_communication = True

        self.times = {}
        self.times['forward'] = 0
        self.times['backward'] = 0
        self.times['comm_grad'] = 0
        self.times['calc_kfgrad'] = 0
        self.times['update_time'] = 0
        self.times['cov_ema_update_time'] = 0

    lr = optimizer.HyperparameterProxy('lr')

    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam)

    def update(self, lossfun=None, *args, **kwds):
        if self.inv_server:
            comm = self.communicator
            if comm is not None and hasattr(comm, 'is_master'):
                if comm.is_master:
                    self.recv_cov_ema()
                    self.inv_update()
                    self.send_inv()
                else:
                    if self.t % self.hyperparam.inv_freq and self.t > 0:
                        self.send_cov_ema()
                        self.recv_inv()
                    else:
                        self.update_main(lossfun, *args, **kwds)
                        self.cov_ema_update()
        else:
            self.update_main(lossfun, *args, **kwds)
            self.cov_ema_update()
            if self.t % self.hyperparam.inv_freq and self.t > 0:
                self.inv_update()

    def update_main(self, lossfun=None, *args, **kwds):
        update_time = time.time()
        if lossfun is not None:
            self.times['start'] = time.time()
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            self.times['forward'] += time.time() - self.times['start']
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            # We will remove ``loss.backward()`` from here.
            # Do backprop, and obtain ``grads`` which contains the dependency
            # graph inside.
            backward_main = getattr(loss, '_backward_main')

            self.times['start'] = time.time()
            self.acts_dict, self.grads_dict, self.ranks_dict = \
                _kfac_backward(self.target, backward_main)
            del loss  # No more backward computation, free memory
            self.times['backward'] += time.time() - self.times['start']

            self.times['start'] = time.time()
            # ======== communication ========
            if self._require_communication:
                target = self.target
                if self.is_changed(target):
                    # NN changed from previous iteration, must unify weights
                    # within all processes
                    self.communicator.broadcast_data(target)
                    return
                # Sumup all gradients, activations, and gs
                self.communicator.allreduce_grad(target)
            # ===============================
            self.times['comm_grad'] += time.time() - self.times['start']

            def get_param(path):
                for _name, _param in self.target.namedparams():
                    if _name == path:
                        return _param
                return None

            self.times['start'] = time.time()
            for linkname in self.ranks_dict.keys():
                if linkname in self.inv_dict.keys():
                    param_W = get_param(linkname + '/W')
                    param_b = get_param(linkname + '/b')
                    if param_W is None:
                        # Some links has empty b param, only return if W is
                        # None.
                        return
                    grad = param_W.grad
                    if param_b is not None:
                        grad = np.column_stack([grad, param_b.grad])
                    A_inv, G_inv = self.inv_dict[linkname]
                    # TODO CPU/GPU impl
                    kfgrads = np.dot(np.dot(G_inv.T, grad), A_inv)
                    if param_b is not None:
                        param_W.kfgrad = kfgrads[:, :-1]
                        param_b.kfgrad = kfgrads[:, -1]
                    else:
                        param_W.kfgrad = kfgrads
            self.times['calc_kfgrad'] += time.time() - self.times['start']

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()

        self.times['update_time'] += time.time() - update_time

    def cov_ema_update(self):
        cov_ema_update_time = time.time()
        for linkname in self.ranks_dict.keys():
            acts = self.acts_dict[linkname]
            grads = self.grads_dict[linkname]
            if acts.ndim == 2:
                A, G = cov_linear(acts, grads)
            elif acts.ndim == 4:
                A, G = cov_conv2d(acts, grads, None)
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    acts.shape))

            # ======== communication ========
            if self._require_communication:
                A_link = DummyLink(A)
                G_link = DummyLink(G)
                self.communicator.allreduce_grad(A_link)
                self.communicator.allreduce_grad(G_link)
                A = A_link.unpack()
                G = G_link.unpack()
            # ===============================

            alpha = self.hyperparam.cov_ema_decay
            if linkname in self.cov_ema_dict.keys():
                # Update EMA of covariance matrices
                A_ema, G_ema = self.cov_ema_dict[linkname]
                A_ema = alpha * A + (1 - alpha) * A_ema
                G_ema = alpha * G + (1 - alpha) * G_ema
                self.cov_ema_dict[linkname] = (A_ema, G_ema)
            else:
                self.cov_ema_dict[linkname] = (A, G)

        self.times['cov_ema_update_time'] += time.time() - cov_ema_update_time


    def inv_update(self):
        for linkname, (A_ema, G_ema) in self.cov_ema_dict.items():
            A_dmp = np.identity(A_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            G_dmp = np.identity(G_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            A_inv = np.linalg.inv(A_ema + A_dmp)
            G_inv = np.linalg.inv(G_ema + G_dmp)
            self.inv_dict[linkname] = (A_inv, G_inv)

    def is_changed(self, target):
        previous_params = self.target_params
        self.target_params = [(name, param.data is not None)
                              for name, param in sorted(target.namedparams())]
        if len(previous_params) != len(self.target_params):
            return True

        for param1, param2 in zip(self.target_params, previous_params):
            if (param1[0] != param2[0]) or param1[1] != param2[1]:
                return True
        return False