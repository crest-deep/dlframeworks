import chainer
from chainer import training
from chainer import optimizer
from chainer.backends import cuda
from chainer.functions import im2col
import numpy as np

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 1
_default_hyperparam.damping = 0.001

_linear_function = chainer.functions.connection.linear.LinearFunction
_convolution_2d_function = chainer.functions.connection.convolution_2d.Convolution2DFunction

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

    def __init__(self, arr):
        self._params = {}
        for name, data in arr.items():
            self._params[name] = DummyVariable(data)

    def namedparams(self):
        for name, param in self._params.items():
            yield name, param

    def unpack(self):
        arr = {}
        for name, param in self._params.items():
            arr[name] = param.data
        return arr


class DummyVariable(object):
    """A dummy variable that returns ``data`` at ``grad``.

    Similar to ``DummyLink``, this class is a wrapper class the behaves like
    ``chainer.Variable``.

    """

    def __init__(self, data):
        self._check(data)
        self._data = [data]

    def _check(self, data):
        if (data is not None and
                not isinstance(data, chainer.get_array_types())):
            msg = '''numpy.ndarray or cuda.ndarray are expected.
Actual: {0}'''.format(type(data))
            raise TypeError(msg)

    @property
    def data(self):
        return self._data[0]

    @property
    def grad(self):
        return self._data[0]

    @grad.setter
    def grad(self, data):
        self._check(data)
        self.grad = data

# TODO CPU/GPU impl
def _cov_linear(acts, grads):
    acts = cuda.to_cpu(acts)
    n, _ = acts.shape
    ones = np.ones(n)
    acts_plus = np.column_stack((acts, ones))
    A = acts_plus.T.dot(acts_plus) / n
    G = grads.T.dot(grads) / n
    return A, G

# TODO CPU/GPU impl
def _cov_conv2d(acts, grads, ksize, stride, pad):
    acts = cuda.to_cpu(acts)
    acts_expand = im2col(acts, ksize, stride, pad).data
    n, _, ho, wo = acts_expand.shape
    acts_expand = acts_expand.transpose(0, 2, 3, 1)
    acts_expand = acts_expand.reshape(n*ho*wo, -1)
    ones = np.ones(n*ho*wo)
    acts_expand_plus = np.column_stack((acts_expand, ones))
    A = acts_expand_plus.T.dot(acts_expand_plus) / n

    n, _, ho, wo = grads.shape
    grads = grads.transpose(0, 2, 3, 1)
    grads = grads.reshape(n*ho*wo, -1)
    G = grads.T.dot(grads) / (n*ho*wo)
    return A, G

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
    conv_args_dict = {}
    for node, grads in grads.items():
        creator_node = node.creator_node  # parent function node
        if creator_node is not None:  # ignore leaf node
            if isinstance(creator_node, _linear_function) \
              or isinstance(creator_node, _convolution_2d_function):
                (acts, param) = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                assert linkname is not None, 'linkname cannot be None.' 
                acts_dict[linkname] = acts.data  # numpy or cupy
                grads_dict[linkname] = grads.data  # numpy or cupy
                ranks_dict[linkname] = creator_node.rank
                if isinstance(creator_node, _convolution_2d_function):
                  conv = creator_node
                  stride, pad  = conv.sy, conv.ph
                  _, _, ksize, _ = param.data.shape
                  conv_args_dict[linkname] = ksize, stride, pad
    return acts_dict, grads_dict, ranks_dict, conv_args_dict


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

    def __init__(self, 
                 communicator=None,
                 lr=_default_hyperparam.lr,
                 cov_ema_decay=_default_hyperparam.cov_ema_decay,
                 inv_freq=_default_hyperparam.inv_freq,
                 damping=_default_hyperparam.damping):
        super(KFAC, self).__init__()
        self.communicator = communicator
        self.hyperparam.lr = lr
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.target_params = []
        self.acts_dict = {}
        self.grads_dict = {}
        self.ranks_dict = {}
        self.conv_args_dict = {}
        self.cov_ema_dict = {}
        self.inv_dict = {}

    lr = optimizer.HyperparameterProxy('lr')

    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam)

    def update(self, lossfun=None, *args, **kwds):
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            # We will remove ``loss.backward()`` from here.
            # Do backprop, and obtain ``grads`` which contains the dependency
            # graph inside.
            backward_main = getattr(loss, '_backward_main')

            self.acts_dict, self.grads_dict, self.ranks_dict, self.conv_args_dict = \
                _kfac_backward(self.target, backward_main)
            del loss  # No more backward computation, free memory

            # ======== communication ========
            if self.communicator is not None:
                target = self.target
                a_s_link = DummyLink(self.acts_dict)
                g_s_link = DummyLink(self.grads_dict)
                if self.is_changed(target):
                    # NN changed from previous iteration, must unify weights
                    # within all processes
                    self.communicator.broadcast_data(target)
                    return
                # Sumup all gradients, activations, and gs
                self.communicator.allreduce_grad(target)
                self.communicator.allreduce_grad(a_s_link)
                self.communicator.allreduce_grad(g_s_link)
                self.acts_dict = a_s_link.unpack()
                self.grads_dict = g_s_link.unpack()
            # ===============================

            def get_param(path):
                for _name, _param in self.target.namedparams():
                    if _name == path:
                        return _param
                return None

            for linkname in self.ranks_dict.keys():
                if linkname in self.inv_dict.keys():
                    param_W = get_param(linkname + '/W')
                    param_b = get_param(linkname + '/b')
                    if param_W is None:
                        # Some links has empty b param, only return if W is
                        # None.
                        return
                    grad = param_W.grad
                    A_inv, G_inv = self.inv_dict[linkname]
                    if param_b is not None:
                        grad = np.column_stack([grad, param_b.grad])
                    else:
                        A_inv = A_inv[:-1, :-1]
                    # TODO CPU/GPU impl
                    grad = cuda.to_cpu(grad)
                    param_shape = grad.shape
                    grad = grad.reshape(param_shape[0], -1)
                    kfgrads = np.dot(np.dot(G_inv.T, grad), A_inv)
                    kfgrads = kfgrads.reshape(param_shape)
                    if param_b is not None:
                        param_W.kfgrad = kfgrads[:, :-1]
                        param_b.kfgrad = kfgrads[:, -1]
                    else:
                        param_W.kfgrad = kfgrads

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()

    def cov_ema_update(self):
        for linkname in self.ranks_dict.keys():
            acts = self.acts_dict[linkname]
            grads = self.grads_dict[linkname]
            if acts.ndim == 2: # linear
                A, G = _cov_linear(acts, grads)
            elif acts.ndim == 4: # convolution_2d
                ksize, stride, pad = self.conv_args_dict[linkname] 
                A, G = _cov_conv2d(acts, grads, ksize, stride, pad)
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    acts.shape))
            alpha = self.hyperparam.cov_ema_decay
            if linkname in self.cov_ema_dict.keys():
                # Update EMA of covariance matrices
                A_ema, G_ema = self.cov_ema_dict[linkname]
                A_ema = alpha * A + (1 - alpha) * A_ema
                G_ema = alpha * G + (1 - alpha) * G_ema
                self.cov_ema_dict[linkname] = (A_ema, G_ema)
            else:
                self.cov_ema_dict[linkname] = (A, G)

    def inv_update(self):
        for linkname, emas in self.cov_ema_dict.items():
            self.inv_update_core(linkname, emas)

    def inv_update_core(self, linkname, emas):
        with cuda.get_device_from_array(emas) as dev:
            if int(dev) == -1:
                self.inv_update_core_cpu(linkname, emas)
            else:
                self.inv_update_core_gpu(linkname, emas)
                
    def inv_update_core_cpu(self, linkname, emas):
        def inv(ema):
            dmp = np.identity(ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            return np.linalg.inv(ema + dmp)
        invs = (inv(ema) for ema in emas)
        self.inv_dict[linkname] = invs

    def inv_update_core_gpu(self, linkname, ema):
        # TODO GPU Impl.
        raise NotImplementedError

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

