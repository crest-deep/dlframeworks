import chainer
from chainer import training
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
            if getattr(creator_node, '_input_indexes_to_retain') is not None:
                a, param = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                if linkname is not None:
                    # params that its linkname is None, are output layer (e.g.
                    # softmax layer). These layers do not have laernable
                    # param inside.
                    data[linkname] = (creator_node.rank, a.data, grad.data)
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
        kfgrad = param.kfgrad
        if kfgrad is None:
            return
        cuda.elementwise('T kfgrad, T lr', 'T param',
                         'param -= lr * kfgrad',
                         'ngd')(kfgrad, self.hyperparam.lr, param.data)


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

        self.act_grad_dict = {}
        self.cov_ema_dict = {}
        self.inv_dict = {}

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
            # We will comment ``loss.backward()`` and call custom backward
            # function to enable KFAC.
            # loss.backward(loss_scale=self._loss_scale)
            backward_main = getattr(loss, '_backward_main')
            self.act_grad_dict = _kfac_backward(self.target, backward_main)
            del loss

            def get_param(path):
                for _name, _param in self.target.namedparams():
                    if _name == path:
                        return _param
                return None

            for linkname in self.act_grad_dict.keys():
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
                    # TODO change for CPU/GPU impl
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
        for linkname, (rank, a, g) in self.act_grad_dict.items():
            if a.ndim == 2:
                mz, _ = a.shape
            elif a.ndim == 4:
                _, _, mz, _ = a.shape
            else:
                raise ValueError('Invalid or unsupported shape: {}.'.format(
                    a.shape))
            ones = np.ones(mz)
            a_plus = np.column_stack((a, ones))
            A = a_plus.T.dot(a_plus) / mz
            G = g.T.dot(g) / mz
            alpha = self.hyperparam.cov_ema_decay
            if linkname in self.cov_ema_dict.keys():
                A_ema, G_ema = self.cov_ema_dict[linkname]
                A_ema = alpha * A + (1 - alpha) * A_ema
                G_ema = alpha * G + (1 - alpha) * G_ema
                self.cov_ema_dict[linkname] = (A_ema, G_ema)
            else:
                self.cov_ema_dict[linkname] = (A, G)

    def inv_update(self):
        for linkname, (A_ema, G_ema) in self.cov_ema_dict.items():
            A_dmp = np.identity(A_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            G_dmp = np.identity(G_ema.shape[0]) * \
                np.sqrt(self.hyperparam.damping)
            A_inv = np.linalg.inv(A_ema + A_dmp)
            G_inv = np.linalg.inv(G_ema + G_dmp)
            self.inv_dict[linkname] = (A_inv, G_inv)
