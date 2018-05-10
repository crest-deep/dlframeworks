import chainer
from chainer import optimizer
from chainer.backends import cuda

from dlframeworks.chainer.utils import get_divided_linknames
from dlframeworks.chainer.optimizers import fisher_block

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.001
_default_hyperparam.momentum = 0.9
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.inv_freq = 20
_default_hyperparam.damping = 0.001

_target_functions = [
    chainer.functions.connection.linear.LinearFunction,
    chainer.functions.connection.convolution_2d.Convolution2DFunction,
    chainer.functions.normalization.batch_normalization.BatchNormalization
    ]

_linear_link = \
    chainer.links.connection.linear.Linear
_convolution_2d_link = \
    chainer.links.connection.convolution_2d.Convolution2D
_batch_norm_link = \
    chainer.links.normalization.batch_normalization.BatchNormalization


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
        fisher_blocks (dict): Keep data to compute Fisher block.

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

        self.fisher_blocks = {}
        self.inv_alg = inv_alg

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')
    cov_ema_decay = optimizer.HyperparameterProxy('cov_ema_decay')
    inv_freq = optimizer.HyperparameterProxy('inv_freq')
    damping = optimizer.HyperparameterProxy('damping')

    def setup(self, link):
        super(KFAC, self).setup(link)
        for linkname, sub_link in link.namedlinks():
            if isinstance(sub_link, _linear_link):
                self.fisher_blocks[linkname] = \
                    fisher_block.FisherBlockLinear(sub_link, linkname)
            elif isinstance(sub_link, _convolution_2d_link):
                self.fisher_blocks[linkname] = \
                    fisher_block.FisherBlockConv2D(sub_link, linkname)
            elif isinstance(sub_link, _batch_norm_link):
                self.fisher_blocks[linkname] = \
                    fisher_block.FisherBlockBatchNorm(sub_link, linkname)
            else:
                continue
        return self

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

            # We removed ``loss.backward()`` from here.
            # Do backprop, and obtain ``grads`` which contains the dependency
            # graph inside.
            backward_main = getattr(loss, '_backward_main')
            self.kfac_backward(self.target, backward_main)

            del loss  # No more backward computation, free memory

            # Update param.kfgrad for each layer
            self.kfgrad_update()

        self.reallocate_cleared_grads()
        self.call_hooks('pre')

        self.t += 1
        for param in self.target.params():
            param.update()

        self.reallocate_cleared_grads()
        self.call_hooks('post')

        self.cov_ema_update()

        comm = self.communicator
        if comm is not None:
            comm.reduce_scatterv(self.target, self.cov_ema_dict)
        if self.t % self.hyperparam.inv_freq == 0 and self.t > 0:
            self.inv_update()
        if comm is not None:
            comm.allgatherv(self.target)

    def kfac_backward(self, link, backward_main):
        """Backward function for KFAC optimizer.
        This function is invoked from ``KFAC.update()`` to:
            1. calculate backprop
            2. obtain the following data for each layer (`~chainer.link.Link`)
                - acts (inputs = activations after previous layer)
                - grads (gradients of outputs)
                - rank (`~chainer.FunctionNode.rank`)
                - conv2d_args (arguments of `~chainer.links.connection.\
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

        for node, out_grads_var in grads.items():
            creator_node = node.creator_node  # parent function node
            if creator_node is not None:  # ignore leaf node
                if not any([isinstance(creator_node, t)
                            for t in _target_functions]):
                    continue
                (in_acts_var, param) = creator_node.get_retained_inputs()
                linkname = get_linkname(param)
                fb = self.fisher_blocks[linkname]
                fb.load_data(in_acts_var.data, out_grads_var.data)
                fb.load_conv2d_args(creator_node, param)

    def kfgrad_update(self):
        """Update param.kfgrad which used for K-FAC updates for each laeyer.
        """
        for fb in self.fisher_blocks.values():
            fb.update_kfgrads()

    def cov_ema_update(self):
        """Update EMA of covariance for each laeyer.
        """
        for fb in self.fisher_blocks.values():
            fb.update_cov_emas(alpha=self.hyperparam.cov_ema_decay)

    def inv_update(self):
        """Update inverse of EMA of covariance for each laeyer.
        """
        comm = self.communicator
        if comm is not None:
            divided_linknames = get_divided_linknames(self.target, comm.size)
            local_linknames = divided_linknames[comm.rank]
            fisher_blocks = [self.fisher_blocks[linkname]
                             for linkname in local_linknames]
        else:
            fisher_blocks = self.fisher_blocks
        for fb in fisher_blocks.values():
            fb.update_invs(damping=self.hyperparam.damping)
