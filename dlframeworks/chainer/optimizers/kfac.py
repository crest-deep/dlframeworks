import chainer
from chainer import training
from chainer.backends import cuda
from chainer.dataset import convert

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

    def param_exists(funcnode):
        for input_varnode in funcnode.inputs:
            for name, param in namedparams:
                if input_varnode.get_variable() is param:
                    return True, name, param
        return False, None, None

    data = {}
    for node, grad in grads.items():
        creator_node = node.creator_node  # parent function node
        if creator_node is not None:  # ignore leaf node
            exists, name, param = param_exists(creator_node)
            if exists:
                a = param
                g = grad
                rank = creator_node.rank
                data[name] = (rank, a.data, g.data)
    sorted(data, key=lambda x: x[0])  # sort by rank
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

    def update(self, lossfun=None, *args, **kwargs):
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
            for link_name, link in self.target.namedlinks():
                if link_name in self.inv_dict.keys():
                    params = list(link.params())  # all parameters under link
                    # Return if there is a NoneType parameter
                    if None in params:
                        return
                    # Assumptions:
                    #   - all grad in grads has same number of rows
                    #   - all grad in grads is 2-dim array
                    offsets = []
                    grads = []
                    for param in params:
                        offsets.append(len(param[0]))
                        offsets.append()
                    grads = [param.grad for param in params]
                    merged_grad = np.column_stack(grads)
                    A_inv, G_inv = self.inv_dict[link_name]
                    # TODO(Yohei) change for CPU/GPU implementation.
                    kfgrads = np.dot(np.dot(G_inv.T, merged_grad), A_inv)
                    for param in params:
                        param.kfgrad = 
            # ================================

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()


    def cov_ema_update(self):
        for (acts, grads), link_name in self.act_grad_dict.items():
            mz, _ = acts.shape
            ones = np.ones(mz)
            acts_plus = np.column_stack((acts, ones))
            A = acts_plus.T.dot(acts_plus) / mz
            G = grads.T.dot(grads) / mz
            alpha = self.hyperparam.cov_ema_decay
            if link_name in self.cov_ema_dict.keys():
                A_ema, G_ema = self.cov_ema_dict[link_name]
                A_ema = alpha * A + (1 - alpha) * A_ema
                G_ema = alpha * G + (1 - alpha) * G_ema
                self.cov_ema_dict[link_name] = (A_ema, G_ema)
            else:
                self.cov_ema_dict[link_name] = (A, G)

    def inv_update(self):
        for (A_ema, G_ema), link_name in self.cov_ema_dict.items():
            A_dmp = np.identity(A_ema.shape[0]) * math.sqrt(self.hyperparam.damping)
            G_dmp = np.identity(G_ema.shape[0]) * math.sqrt(self.hyperparam.damping)
            A_inv = np.linalg(A_ema + A_dmp)
            G_inv = np.linalg(G_ema + G_dmp)
            self.inv_dict[link_name] = (A_inv, G_inv)
