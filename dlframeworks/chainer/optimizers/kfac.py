import chainer
from chainer import optimizer
from chainer.backends import cuda
from chainer.functions import im2col
import numpy as np
import cupy

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 1
_default_hyperparam.damping = 0.001


_linear_function = chainer.functions.connection.linear.LinearFunction
_convolution_2d_function = chainer.functions.connection.convolution_2d.Convolution2DFunction

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
    acts_expand = _acts_expand_convolution_2d(acts, ksize, stride, pad)
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
    lib = np if int(dev) == -1 else cupy
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


def _acts_expand_convolution_2d(acts, ksize, stride, pad):
    acts_expand = im2col(acts, ksize, stride, pad).data
    n, c, ho, wo = acts_expand.shape
    acts_expand = acts_expand.transpose(0, 2, 3, 1)
    acts_expand = acts_expand.reshape(n*ho*wo, -1)
    return acts_expand


def _kfac_backward(loss, chain):
    namedparams = list(chain.namedparams())

    def get_linkname(param):
        # Get a linkname from a parameter.
        for _name, _param in namedparams:
            if param is _param:
                # Only return linkname NOT paramname.
                return _name[:_name.rfind('/')]
        return None

    acts_dict = {}
    grads_dict = {}
    rank_dict = {}
    conv_args_dict = {}

    output_vars = []
    linknames = []

    func_node = loss.creator_node
    while func_node:
        if isinstance(func_node, _linear_function) \
          or isinstance(func_node, _convolution_2d_function):
            (acts_var, param) = func_node.get_retained_inputs()
            linkname = get_linkname(param)
            assert linkname is not None, 'linkname cannot be None.'
            acts_dict[linkname] = acts_var.data  
            rank_dict[linkname] = func_node.rank
            linknames.append(linkname)

            (preacts_var,) = func_node.get_retained_outputs()
            output_vars.append(preacts_var)

            if isinstance(func_node, _convolution_2d_function):
                conv = func_node
                stride, pad = conv.sy, conv.ph
                _, _, ksize, _ = param.data.shape
                conv_args_dict[linkname] = ksize, stride, pad

        input_var_node = func_node.inputs[0]
        func_node = input_var_node.creator_node

    grads_vars = chainer.grad([loss], output_vars)
    for i, linkname in enumerate(linknames):
        grads_dict[linkname] = grads_vars[i].data

    return acts_dict, grads_dict, rank_dict, conv_args_dict


def _kfac_grad_update(dev, param_W, param_b, invs):
    A_inv, G_inv = invs
    grad = param_W.grad
    if grad.ndim == 4: # convolution_2d
        c_o, c_i, h, w = grad.shape
        grad = grad.reshape(c_o, -1)
    if param_b is not None:
        if int(dev) == -1:
            grad = np.column_stack([grad, param_b.grad])
        else:
            grad = cupy.column_stack([grad, param_b.grad])
    kfgrads = (G_inv.T.dot(grad)).dot(A_inv).astype(grad.dtype)
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
        d0, d1, d2 = array.shape
        if index == 0:
            array = array.reshape(d0, d1*d2)
            return inv.dot(array).reshape(d0, d1, d2)    
        elif index == 1:
            array = array.transpose(1, 0, 2)
            array = array.reshape(d1, d0*d2)
            result = inv.dot(array).reshape(d1, d0, d2)    
            return result.transpose(1, 0, 2)
        elif index == 2:
            array = array.transpose(2, 0, 1)
            array = array.reshape(d2, -1)
            result = inv.dot(array).reshape(d2, d0, d1)    
            return result.transpose(2, 0, 1)
        else:
            raise ValueError('Index has to be in [0, 1, 2]')

    kfgrad = rmatmul(G_inv, rmatmul(V_inv, rmatmul(U_inv, grad)))
    param_W.kfgrad = kfgrad.reshape(param_W.grad.shape)


class KFACUpdateRule(chainer.optimizer.UpdateRule):

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
        cuda.elementwise('T grad, T lr, T momentum', 
                         'T param, T v',
                         '''v = momentum * v - lr * grad;
                            param += v;''',
                         'ngd')(
                             grad, self.hyperparam.lr, self.hyperparam.momentum,
                             param.data, self.state['v'])


class KFAC(chainer.optimizer.GradientMethod):

    def __init__(self, 
                 communicator=None,
                 lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum,
                 cov_ema_decay=_default_hyperparam.cov_ema_decay,
                 inv_freq=_default_hyperparam.inv_freq,
                 damping=_default_hyperparam.damping,
                 use_doubly_factored=True,):
        super(KFAC, self).__init__()
        self.communicator = communicator
        self.inv_server = inv_server
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.use_doubly_factored = use_doubly_factored
        self.target_params = []
        self.acts_dict = {}
        self.grads_dict = {}
        self.rank_dict = {}
        self.conv_args_dict = {}
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
    momentum = optimizer.HyperparameterProxy('momentum')


    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam)


    def update(self, lossfun=None, *args, **kwds):
        if lossfun is not None:
            self.times['start'] = time.time()
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            self.times['forward'] += time.time() - self.times['start']
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()

            loss.backward()
            del loss  # No more backward computation, free memory
            self.times['backward'] += time.time() - self.times['start']

            for linkname, invs in self.inv_dict.items():
                param_W = self.get_param(linkname + '/W')
                param_b = self.get_param(linkname + '/b')
                # Some links has empty b param
                assert param_W is not None
                data = (param_W.data, param_b.data, invs) \
                    if param_b is not None else (param_W.data, invs)

                with cuda.get_device_from_array(*data) as dev:
                    if len(invs) >= 3:
                        _kfac_grad_update_doubly_factored(param_W, param_b, invs)
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


    def cov_ema_update(self, lossfun=None, *args, **kwds):
        if lossfun is not None:
            loss = lossfun(*args, **kwds)
            self.acts_dict, self.grads_dict, self.rank_dict, self.conv_args_dict = \
                _kfac_backward(loss, self.target)

            for linkname in self.rank_dict.keys():
                self.cov_ema_update_core(linkname)


    def cov_ema_update_core(self, linkname):
        acts = self.acts_dict[linkname]
        grads = self.grads_dict[linkname]
        nobias = self.get_param(linkname + '/b') is None
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


        self.times['cov_ema_update_time'] += time.time() - cov_ema_update_time

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
