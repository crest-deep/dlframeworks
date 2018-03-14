import chainer
from chainer import training
from chainer import optimizer
from chainer.backends import cuda
from chainer.dataset import convert
from collections import OrderedDict
import numpy as np

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 20
_default_hyperparam.damping = 0.001


def _kfac_backward(link, backward_main, retain_grad=True,
                   enable_double_backprop=False, loss_scale=None):
    """Backward function for KFAC optimizer.

    This function is invoked from ``KFAC.update()`` to caculate the gradients.
    KFAC needs the inputs and gradients per layer, and Chainer does not let us
    get these objects directly from the API.

    """
    with chainer.using_config('enable_backprop', enable_double_backprop):
        # To obtain grads, we need to edit the origianl file (`variable.py`)
        grads = backward_main(retain_grad, loss_scale)

    namedparams = list(link.namedparams())

    def get_linkname(param):
        # Get a linkname from a parameter.
        for _name, _param in namedparams:
            if param is _param:
                # Only return linkname NOT paramname.
                return _name[:_name.rfind('/')]
        return None

    data = {}
    for node, grad in grads.items():
        creator_node = node.creator_node  # parent function node
        if creator_node is not None:  # ignore leaf node
            if isinstance(creator_node, chainer.functions.connection.linear.LinearFunction) \
              or isinstance(creator_node, chainer.functions.connection.convolution_2d.Convolution2DFunction):
                (a, param) = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                assert linkname is not None, 'linkname cannot be None.'
                data[linkname] = (creator_node.rank, a.data, grad.data, param.data.shape)
    # Sort by its rank
    data = OrderedDict(sorted(data.items(), key=lambda x: x[1][0]))
    return data


class KFACUpdater(training.updaters.StandardUpdater):

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 models=None, device=None, loss_func=None, loss_scale=None):
        assert isinstance(optimizer, KFAC), \
            'The optimizer has to be an instance of KFAC.'
        super(KFACUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            device=device,
            loss_func=loss_func,
            loss_scale=loss_scale,
        )

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)
        optimizer.cov_ema_update()
        if self.iteration % optimizer.hyperparam.inv_freq == 0 and \
                self.iteration > 0:
            optimizer.inv_update()


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
                 lr=_default_hyperparam.lr,
                 cov_ema_decay=_default_hyperparam.cov_ema_decay,
                 inv_freq=_default_hyperparam.inv_freq,
                 damping=_default_hyperparam.damping):
        super(KFAC, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.inv_freq = inv_freq
        self.hyperparam.damping = damping

        self.data_dict = {}
        self.cov_ema_dict = {}
        self.inv_dict = {}

    lr = optimizer.HyperparameterProxy('lr')

    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam)

    def update(self, lossfun=None, *args, **kwds):
        print ('update')
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            # We will comment ``loss.backward()`` and call custom backward
            # function to enable KFAC.
            # loss.backward(loss_scale=self._loss_scale)
            backward_main = getattr(loss, '_backward_main')
            self.data_dict = _kfac_backward(self.target, backward_main)
            del loss

            def get_param(path):
                for _name, _param in self.target.namedparams():
                    if _name == path:
                        return _param
                return None

            for linkname in self.data_dict.keys():
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
            # ================================

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()

    def cov_ema_update(self):
        print ('cov_ema_update')
        # TODO CPU/GPU impl
        def cov_linear(a, g):
            print('cov_linear')
            N, _ = a.shape
            ones = np.ones(N)
            a_plus = np.column_stack((a, ones))
            A = a_plus.T.dot(a_plus) / N
            G = g.T.dot(g) / N
            return A, G

        # TODO CPU/GPU impl
        def cov_conv2d(a, g, param_shape):
            print('cov_conv2d')
            N, J, H, W = a.shape
            I, J, H_k, W_k = param_shape
            T = H * W     # number of spatial location in an input feature map
            D = H_k * W_k # number of spatial location in a kernel
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
                          print ('n{0} j{1} h_{2} w_{3}'.format(n,j,h_,w_))
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

        for linkname, (rank, a, g, param_shape) in self.data_dict.items():
            if a.ndim == 2:
                A, G = cov_linear(a, g)
            elif a.ndim == 4:
                A, G = cov_conv2d(a, g, param_shape)
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    a.shape))
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
        print ('inv_update')
        for linkname, (A_ema, G_ema) in self.cov_ema_dict.items():
            A_dmp = np.identity(A_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            G_dmp = np.identity(G_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            A_inv = np.linalg.inv(A_ema + A_dmp)
            G_inv = np.linalg.inv(G_ema + G_dmp)
            self.inv_dict[linkname] = (A_inv, G_inv)

