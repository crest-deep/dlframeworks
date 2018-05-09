import collections

import chainer
from chainer import optimizer
from chainer.backends import cuda
from chainer.functions import im2col

from dlframeworks.chainer.utils import get_divided_linknames

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.001
_default_hyperparam.momentum = 0.9
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 20
_default_hyperparam.damping = 0.001

_linear_function = \
    chainer.functions.connection.linear.LinearFunction
_linear_link = \
    chainer.links.connection.linear.Linear
_convolution_2d_function = \
    chainer.functions.connection.convolution_2d.Convolution2DFunction
_convolution_2d_link = \
    chainer.links.connection.convolution_2d.Convolution2D


def _cov_linear(xp, acts, grads, nobias):
    # Note that this method is called inside a with-statement of xp module
    n, _ = acts.shape
    if not nobias:
        ones = xp.ones(n)
        acts = xp.column_stack((acts, ones))
    A = acts.T.dot(acts) / n
    G = grads.T.dot(grads) / n
    return [A, G]


def _cov_convolution_2d(xp, acts, grads, nobias, ksize, stride, pad):
    # Note that this method is called inside a with-statement of xp module
    n, _, _, _ = acts.shape
    acts_expand = _acts_expand_convolution_2d(acts, ksize, stride, pad)
    if not nobias:
        ones = xp.ones(acts_expand.shape[0])
        acts_expand = xp.column_stack((acts_expand, ones))
    A = acts_expand.T.dot(acts_expand) / n
    n, _, ho, wo = grads.shape
    grads = grads.transpose(0, 2, 3, 1)
    grads = grads.reshape(n*ho*wo, -1)
    G = grads.T.dot(grads) / (n*ho*wo)
    return [A, G]


def _acts_expand_convolution_2d(acts, ksize, stride, pad):
    acts_expand = im2col(acts, ksize, stride, pad).data
    # n x c*ksize*ksize x ho x wo
    n, _, ho, wo = acts_expand.shape
    # n x ho x wo x c*ksize*ksize
    acts_expand = acts_expand.transpose(0, 2, 3, 1)
    # n*ho*wo x c*ksize*ksize
    acts_expand = acts_expand.reshape(n*ho*wo, -1)
    return acts_expand


class KFACUpdateRule(chainer.optimizer.UpdateRule):

    """Update rule for K-FAC.

    See :class:`~chainer.optimizers.KFAC` for the default value of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(KFACUpdateRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        v = self.state['v']
        v *= self.hyperparam.momentum
        v -= self.hyperparam.lr * grad
        param.data += v

    def update_core_gpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
            param += v;''',
            'kfac')(grad, self.hyperparam.lr, self.hyperparam.momentum,
                    param.data, self.state['v'])


class KFAC(chainer.optimizer.GradientMethod):

    """K-FAC optimizer.

    See: `Optimizing Neural Networks with \
          Kronecker-factored Approximate Curvature \
          <https://arxiv.org/abs/1503.05671>`_

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.
        cov_ema_decay (float): Decay factor used when calculating the
                               covariance estimate Exponential Moving Average.
        inv_freq (int): Frequency to calculate the inverse of covariance
                        estimate EMA for each layer.
        inv_alg (str): Algorithm used when calculating the inverse.
        damping (float): Damping factor used to stabilize training
                         due to errors in the local approximation with the
                         Fisher information matrix.

    Attributes:
        acts_dict (dict): Keep inputs (activations after previous layer)
                          for each layer.
        grads_dict (dict): Keep gradients of outputs for each layer.
        rank_dict (dict): Keep `~chainer.FunctionNode.rank` for each layer.
        conv_args_dict (dict): Keep arguments for each convolutional layer
                               (`~chainer.links.connection.\
                                 convolution_2d.Convolution2D`).

    """

    def __init__(self,
                 communicator=None,
                 inv_server=None,
                 lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum,
                 cov_ema_decay=_default_hyperparam.cov_ema_decay,
                 inv_freq=_default_hyperparam.inv_freq,
                 inv_alg=None,
                 damping=_default_hyperparam.damping,):
        super(KFAC, self).__init__()
        self.communicator = communicator
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.acts_dict = {}
        self.grads_dict = {}
        self.rank_dict = {}
        self.conv_args_dict = {}
        self.inv_alg = inv_alg

        self.is_done = False
        self.times = collections.defaultdict(lambda: 0)

        # TODO Initialize below with all batch
        self.cov_ema_dict = {}
        self.inv_dict = {}

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')
    cov_ema_decay = optimizer.HyperparameterProxy('cov_ema_decay')
    inv_freq = optimizer.HyperparameterProxy('inv_freq')
    damping = optimizer.HyperparameterProxy('damping')

    def setup(self, link):
        super(KFAC, self).setup(link)
        self.t_inv = 0
        self.t_cov = 0
        linknames = []
        for linkname, sub_link in link.namedlinks():
            if isinstance(sub_link, _linear_link):
                linknames.append(linkname)
            elif isinstance(sub_link, _convolution_2d_link):
                linknames.append(linkname)
            else:
                continue
        self.linknames = sorted(linknames)

    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam)

    def update(self, lossfun=None, *args, **kwds):
        comm = self.communicator
        self.grad_update(lossfun, *args, **kwds)
        self.cov_ema_update()
        if comm is not None:
            comm.reduce_scatterv(self.target, self.cov_ema_dict)
        if self.t % self.hyperparam.inv_freq == 0 and self.t > 0:
            self.inv_update()
        if comm is not None:
            comm.allgatherv(self.target)

    def grad_update(self, lossfun=None, *args, **kwds):
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()

            # We removed ``loss.backward()`` from here.
            # Do backprop, and obtain ``grads`` which contains the dependency
            # graph inside.
            backward_main = getattr(loss, '_backward_main')
            self._kfac_backward(self.target, backward_main)

            del loss  # No more backward computation, free memory

            # Update param.kfgrad for each layer
            self.kfac_grad_update()

        self.reallocate_cleared_grads()
        self.call_hooks('pre')

        self.t += 1
        for param in self.target.params():
            param.update()

        self.reallocate_cleared_grads()
        self.call_hooks('post')

    def _kfac_backward(self, link, backward_main):
        """Backward function for KFAC optimizer.
        This function is invoked from ``KFAC.grad_update()`` to:
            1. calculate backprop
            2. obtain the following data for each layer (`~chainer.link.Link`)
                - acts (inputs = activations after previous layer)
                - grads (gradients of outputs)
                - rank (`~chainer.FunctionNode.rank`)
                - conv_args (arguments of `~chainer.links.connection.\
                                           convolution_2d.Convolution2D`)
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

        for node, grads in grads.items():
            creator_node = node.creator_node  # parent function node
            if creator_node is not None:  # ignore leaf node
                if isinstance(creator_node, _linear_function) \
                  or isinstance(creator_node, _convolution_2d_function):
                    (acts, param) = creator_node.get_retained_inputs()
                    linkname = get_linkname(param)
                    assert linkname is not None, 'linkname cannot be None.'
                    self.acts_dict[linkname] = acts.data  # numpy or cupy
                    self.grads_dict[linkname] = grads.data  # numpy or cupy
                    self.rank_dict[linkname] = creator_node.rank
                    if isinstance(creator_node, _convolution_2d_function):
                        conv = creator_node
                        stride, pad = conv.sy, conv.ph
                        _, _, ksize, _ = param.data.shape
                        self.conv_args_dict[linkname] = ksize, stride, pad

    def kfac_grad_update(self):
        """Update param.kfgrad which used for K-FAC updates for each laeyer.

        This function refers `self.inv_dict`.
        """
        for linkname, invs in self.inv_dict.items():
            self._kfac_grad_update_core(linkname, invs)

    def _kfac_grad_update_core(self, linkname, invs):
        """Update the value of `~chainer.Parameter.kfgrad`.

        Args:
            linkname (str): Name of link.
            invs (touple): Pair of inverse (A_inv, G_inv).
        """
        param_W = self.get_param(linkname + '/W')
        param_b = self.get_param(linkname + '/b')
        # Some links has empty b param
        assert param_W is not None
        data = (param_W.data, param_b.data, invs) \
            if param_b is not None else (param_W.data, invs)
        xp = cuda.get_array_module(*data)
        with cuda.get_device_from_array(*data):
            A_inv, G_inv = invs
            grad = param_W.grad
            if grad.ndim == 4:  # convolution_2d
                c_o, c_i, h, w = grad.shape
                grad = grad.reshape(c_o, -1)
            if param_b is not None:
                grad = xp.column_stack([grad, param_b.grad])

            kfgrads = xp.dot(xp.dot(G_inv, grad), A_inv).astype(grad.dtype)

            if param_b is not None:
                param_W.kfgrad = kfgrads[:, :-1].reshape(param_W.grad.shape)
                param_b.kfgrad = kfgrads[:, -1].reshape(param_b.grad.shape)
            else:
                param_W.kfgrad = kfgrads.reshape(param_W.grad.shape)

    def get_param(self, path):
        for _name, _param in self.target.namedparams():
            if _name == path:
                return _param
        return None

    def get_link(self, path):
        for _name, _link in self.target.namedlinks():
            if _name == path:
                return _link
        return None

    def allocate_matrices(self):
        dictionary = collections.OrderedDict()
        for linkname in self.linknames:
            link = self.get_link(linkname)
            param_W = self.get_param(linkname + '/W')
            param_b = self.get_param(linkname + '/b')
            if param_W is None:
                raise ValueError('param_W MUST not be None at', linkname)
            xp = cuda.get_array_module(param_W.data)

            with cuda.get_device_from_array(param_W.data):
                if isinstance(link, _linear_link):
                    n_out, n_in = param_W.shape
                    if param_b is not None:
                        A = xp.zeros((n_in + 1, n_in + 1))
                    else:
                        A = xp.zeros((n_in, n_in))
                    G = xp.zeros((n_out, n_out))
                elif isinstance(link, _convolution_2d_link):
                    c_out, c_in, kh, kw = param_W.shape
                    if param_b is not None:
                        A = xp.zeros((c_in*kh*kw + 1, c_in*kh*kw + 1))
                    else:
                        A = xp.zeros((c_in*kh*kw, c_in*kh*kw))
                    G = xp.zeros((c_out, c_out))
                else:
                    continue
            dictionary[linkname] = [A, G]
        return collections.OrderedDict(
            sorted(dictionary.items(), key=lambda x: x[0]))

    def cov_ema_update(self):
        """Update EMA of covariance for each laeyer.

        This function refers `self.rank_dict` to get sorted keys (linknames).
        """
        comm = self.communicator
        if self.t_cov == 0:
            self.cov_ema_dict = self.allocate_matrices()
        # ======== Communication
        if comm is not None:
            is_done = comm.sendrecv_param(self)
            if is_done:
                return True
        for i, linkname in enumerate(sorted(self.rank_dict.keys())):
            self._cov_ema_update_core(linkname)

        # ======== Communication
        if comm is not None:
            comm.sendrecv_cov_ema(self.cov_ema_dict)
            self.t_inv += 1
        self.t_cov += 1

    def _cov_ema_update_core(self, linkname):
        """Update the value of `self.cov_ema_dict[linkname]`.

        Args:
            linkname (str): Key of `self.cov_ema_dict`.
        """
        comm = self.communicator
        acts = self.acts_dict[linkname]
        grads = self.grads_dict[linkname]
        nobias = self.get_param(linkname + '/b') is None
        xp = cuda.get_array_module(acts, grads)
        with cuda.get_device_from_array(acts, grads):
            if acts.ndim == 2:  # linear
                covs = _cov_linear(xp, acts, grads, nobias)
            elif acts.ndim == 4:  # convolution_2d
                ksize, stride, pad = self.conv_args_dict[linkname]
                covs = _cov_convolution_2d(
                        xp, acts, grads, nobias, ksize, stride, pad)
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    acts.shape))
        # ======== Communication
        if comm is not None:
            comm.allreduce_cov(covs)
        if linkname in self.cov_ema_dict.keys():
            alpha = self.hyperparam.cov_ema_decay
            cov_emas = self.cov_ema_dict[linkname]
            for i, cov_ema in enumerate(cov_emas):
                cov_emas[i] = alpha * covs[i] + (1 - alpha) * cov_ema
            self.cov_ema_dict[linkname] = cov_emas
        else:
            self.cov_ema_dict[linkname] = covs

    def inv_update(self):
        """Update inverse of EMA of covariance for each laeyer.

        This function refers `self.cov_ema_dict`.
        """
        comm = self.communicator

        divided_linknames = get_divided_linknames(self.target, comm.size)
        for linkname in divided_linknames[comm.rank]:
            emas = self.cov_ema_dict[linkname]
            self._inv_update_core(linkname, emas)

        self.t_inv += 1

    def _inv_update_core(self, linkname, emas):
        """Update the value of `self.inv_dict[linkname]`.

        Args:
            linkname (str): Key of `self.inv_dict`.
            emas (touple): Pair of EMA of covariance (A_ema, G_ema).
        """
        xp = cuda.get_array_module(*emas)
        with cuda.get_device_from_array(*emas):

            # TODO add plus value (pi) for damping
            def inv_2factors(ema):
                dmp = xp.identity(ema.shape[0]) * \
                  xp.sqrt(self.hyperparam.damping)
                return inv(ema + dmp)

            def inv(X):
                alg = self.inv_alg
                if alg == 'cholesky':
                    c = xp.linalg.inv(xp.linalg.cholesky(X))
                    return xp.dot(c.T, c)
                else:
                    return xp.linalg.inv(X)

            assert len(emas) == 2, 'Length of emas has to be 2.'
            invs = [inv_2factors(ema) for ema in emas]
            self.inv_dict[linkname] = invs
