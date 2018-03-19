import chainer
from chainer import training
from chainer import optimizer
from chainer.backends import cuda
from chainer.functions import im2col
import numpy as np
import cupy

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

    In the communication functions of ChainerMN communicators (e.g.
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

def _cov_linear(dev, acts, grads, nobias):
    n, _ = acts.shape
    if not nobias:
        if int(dev) == -1:
            ones = np.ones(n)
            acts = np.column_stack((acts, ones))
        else:
            ones = cupy.ones(n)
            acts = cupy.column_stack((acts, ones))

    A = acts.T.dot(acts) / n
    G = grads.T.dot(grads) / n
    return [A, G]


def _cov_convolution_2d(dev, acts, grads, nobias, \
                            ksize, stride, pad):
    n, _, _, _ = acts.shape
    acts_expand = _acts_expand_convolution_2d(acts, ksize, strdie, pad)
    if not nobias:
        if int(dev) == -1:
            ones = np.ones(n*ho*wo)
            acts_expand = np.column_stack((acts_expand, ones))
        else:
            ones = cupy.ones(n*ho*wo)
            acts_expand = cupy.column_stack((acts_expand, ones))
    A = acts_expand.T.dot(acts_expand) / n
    G = _grads_cov_convolution_2d(grads)
    return [A, G]
    

def _cov_convolution_2d_doubly_factored(dev, acts, grads, nobias, \
                                            ksize, stride, pad):
    n, _, _, _ = acts.shape
    acts_expand = _acts_expand_convolution_2d(acts, ksize, strdie, pad)
    acts_expand = acts_expand.reshape(n, ho*wo, -1)
    lib = np if int(dev) == -1 else lib = cupy
    u_expand = lib.zeros((n, ho*wo))
    v_expand = lib.zeros((n, c))
    for i in range(n): 
        # TODO implement fast rank-1 approximation
        u, s, v = lib.linalg.svd(acts_expand[i])
        u1 = lib.sqrt(s[0]) * u[0]
        v1 = lib.sqrt(s[0]) * v.T[0]
        u_expand[i] = u1 
        v_expand[i] = v1 
    U = u_expand.T.dot(u_expand) / n
    V = v_expand.T.dot(v_expand) / n
    G = _grads_cov_convolution_2d(grads)
    if nobias:
        return [U, V, G]
    else:
        b_grads = grads.sum(axis=(2,3))
        Fb = b_grads.T.dot(b_grads.T) # full Fisher block for bias
        return [U, V, G, Fb]


def _grads_cov_convolution_2d(grads):
    n, _, ho, wo = grads.shape
    grads = grads.transpose(0, 2, 3, 1)
    grads = grads.reshape(n*ho*wo, -1)
    G = grads.T.dot(grads) / (n*ho*wo)
    return G


def _acts_expand_convolution_2d(acts, ksize, stride, pad)
    acts_expand = im2col(acts, ksize, stride, pad).data
    n, c, ho, wo = acts_expand.shape
    acts_expand = acts_expand.transpose(0, 2, 3, 1)
    acts_expand = acts_expand.reshape(n*ho*wo, -1)
    return acts_expand


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


def _kfac_grad_update(dev, param_W, param_b, invs):
    A_inv, G_inv = invs
    grad = param_W.grad
    c_o, c_i, h, w = grad.shape
    grad = grad.reshape(c_o, -1)
    if param_b is not None:
        if int(dev) == -1:
            grad = np.column_stack([grad, param_b.grad])
        else:
            grad = cupy.column_stack([grad, param_b.grad])
    kfgrads = (G_inv.T.dot(grad)).dot(A_inv).astype(grad.type)
    if param_b is not None:
        param_W.kfgrad = kfgrads[:, :-1].reshape(param_W.grad.shape)
        param_b.kfgrad = kfgrads[:, -1].reshape(param_b.grad.shape)
    else:
        param_W.kfgrad = kfgrads.reshape(param_W.grad.shape)
        
        
def _kfac_grad_update_doubly_factored(param_W, param_b, invs):
    if param_b is not None:
        U_inv, V_inv, G_inv, Fb_inv = invs
         # Apply inverse of full Fisher block (Fb_inv) to bias
         grad = param_b.grad
         kfgrad = Fb_inv.dot(grad)
         param_b.kfgrad = kfgrad
    else:
        U_inv, V_inv, G_inv = invs

    grad = param_W.grad
    c_o, c_i, h, w = grad.shape
    grad = grad.transpose(2, 3, 1, 0)
    grad = grad.reshape(h*w, c_i, c_out)

    def rmatmul(inv, array, index):
        assert array.ndim == 3
        assert index in [0, 1, 2]
        d0, d1, d2 = array.shape
        if index == 0:
            array = array.reshape(d0, d1*d2)
            return inv.dot(array).reshape(d0, d1, d2)    
        elif index == 1:
            array = array.transpose(1, 0, 2)
            array = array.reshape(d1, d0*d2)
            result = inv.dot(array).reshape(d1, d0, d2)    
            return result.transpose(1, 0, 2)
        else index == 2:
            array = array.transpose(2, 0, 1)
            array = array.reshape(d2, -1)
            result = inv.dot(array).reshape(d2, d0, d1)    
            return result.transpose(2, 0, 1)

     kfgrad = rmatmul(G_inv, rmatmul(V_inv, rmatmul(U_inv, grad)))
     param_W.kfgrad = kfgrad.reshape(param_W.grad.shape)


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
                 damping=_default_hyperparam.damping,
                 use_doubly_factored=True,):
        super(KFAC, self).__init__()
        self.communicator = communicator
        self.hyperparam.lr = lr
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.use_doubly_factored = use_doubly_factored
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


            for linkname, invs in self.inv_dict.items():
                param_W = self.get_param(linkname + '/W')
                param_b = self.get_param(linkname + '/b')
                # Some links has empty b param
                assert param_W is not None
                data = (param_W.data, param_b.data, invs) \
                    if param_b is not None else (param_W.data, invs)
                with cuda.get_device_from_array(*data) as dev:
                    if self.use_doubly_factored:
                        _kfac_grad_update_doubly_factored(dev, param_W, param_b, invs)
                    else:
                        _kfac_grad_update(dev, param_W, param_b, invs)

        self.reallocate_cleared_grads()
        self.call_hooks()
        self.t += 1
        for param in self.target.params():
            param.update()


    def get_param(self, path):
        for _name, _param in self.target.namedparams():
            if _name == path:
                return _param
        return None


    def cov_ema_update(self):
        for linkname in self.ranks_dict.keys():
            self.cov_ema_update_core(linkname)


    def cov_ema_update_core(self, linkname):
        acts = self.acts_dict[linkname]
        grads = self.grads_dict[linkname]
        param_b = self.get_param(linkname + '/b')
        nobias = param_b is None
        with cuda.get_device_from_array(acts, grads) as dev:
            if acts.ndim == 2: # linear
                covs = _cov_linear(dev, acts, grads, nobias)
            elif acts.ndim == 4: # convolution_2d
                ksize, stride, pad = self.conv_args_dict[linkname] 
                if self.use_doubly_factored:
                    covs = _cov_convolution_2d(dev, acts, grads, nobias, \
                                               ksize, stride, pad)
                else:
                    covs = _cov_convolution_2d_doubly_factored(
                                               dev, acts, grads, nobias, \
                                               ksize, stride, pad)
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    acts.shape))
            if linkname in self.cov_ema_dict.keys():
                alpha = self.hyperparam.cov_ema_decay
                cov_emas = self.cov_ema_dict[linkname]
                for i, cov_ema in enumerate(cov_emas):
                    cov_emas[i] = alpha * covs[i] + (1 - alpha) * cov_ema
                self.cov_ema_dict[linkname] = cov_emas
            else:
                self.cov_ema_dict[linkname] = covs


    def inv_update(self):
        for linkname, emas in self.cov_ema_dict.items():
            self.inv_update_core(linkname, emas)


    def inv_update_core(self, linkname, emas):
        with cuda.get_device_from_array(*emas) as dev:
            lib = np if int(dev) == -1 else cupy
        num_ema = len(emas)

        def inv_2factors(ema):
            dmp = lib.identity(ema.shape[0]) * \
                lib.sqrt(self.hyperparam.damping)
            return lib.linalg.inv(ema + dmp)

        def inv_3factors(ema):
            dmp = lib.identity(ema.shape[0]) * \
                lib.cbrt(self.hyperparam.damping)
            return lib.linalg.inv(ema + dmp)

        if len(emas) == 2:   # [A_ema, G_ema]
            invs = [inv_2factors(ema) for ema in emas] 
        elif len(emas) == 3: # [U_ema, V_ema, G_ema]
            invs = [inv_3factors(ema) for ema in emas] 
        elif len(emas) == 4: # [U_ema, V_ema, G_ema, Fb_ema]
            invs = [inv_3factors(ema) for ema in emas[:3]]
            Fb_ema = emas[-1]
            dmp = lib.identity(Fb_ema.shape[0]) * \
                               self.hyperparam.damping
            Fb_inv = lib.linalg.inv(Fb_ema + dmp)
            invs.append(Fb_inv)
        else:
            raise ValueError('Lengh of emas has to be in [2, 3, 4]')

        self.inv_dict[linkname] = invs


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

