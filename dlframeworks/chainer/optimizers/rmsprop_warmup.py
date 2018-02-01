import chainer
from chainer.backends import cuda
from chainer import optimizer
import math
import numpy


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01  # Learning rate
_default_hyperparam.alpha_sgd = 0.01  # Learning rate in momentum SGD
_default_hyperparam.mu1 = 0.9  # Momentum in momentum SGD
_default_hyperparam.alpha_rmsprop = 0.01  # Learning rate in RMSProp
_default_hyperparam.mu2 = 0.99  # Alpha in RMSProp
_default_hyperparam.eps = 1e-8  # Epsilon in RMSProp
_default_hyperparam.weight_decay = 0.0005  # Weight deacy rate of LARS


class RMSpropWarmupRule(optimizer.UpdateRule):

    """Update rule for RMSpropWarmup.

    Args:
        lr (float): Learning rate.
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        alpha_sgd (float): Learning rate for momentum SGD.
        mu1 (float): Exponential decay rate of the first order moment.
        alpha_rmsprop (float): Learning rate for RMSProp.
        mu2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        weight_decay (float): Weight decay rate of LARS. Default is off.
        lars (bool): Use LARS or not.

    """

    def __init__(self, parent_hyperparam=None,
                 lr=None,
                 alpha_sgd=None,
                 mu1=None,
                 alpha_rmsprop=None,
                 mu2=None,
                 eps=None,
                 lars=False):
        super(RMSpropWarmupRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.lr = lr
        if alpha_sgd is not None:
            self.alpha_sgd = alpha_sgd
        if mu1 is not None:
            self.mu1 = mu1
        if alpha_rmsprop is not None:
            self.hyperparam.alpha_rmsprop = alpha_rmsprop
        if mu2 is not None:
            self.hyperparam.mu2 = mu2
        if eps is not None:
            self.hyperparam.eps = eps
        self._lars = lars

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))

        # -------- LARS --------
        xp = cuda.get_array_module(param.data)
        if self._lars:
            lr = hp.lr * xp.norm(param.data) / \
                (xp.norm(param.grad) +
                 hp.weight_decay * xp.norm(param.data))
        else:
            lr = hp.lr

        # -------- RMSProp part --------
        ms = self.state['ms']
        ms *= hp.mu2
        ms += (1 - hp.mu2) * grad * grad
        rmsprop_delta = -(hp.alpha_rmsprop * grad / (numpy.sqrt(ms) + eps))

        # -------- Momentum SGD part --------
        v = self.state['v']
        v *= self.hyperparam.mu1
        v -= self.hyperparam.alpha_sgd * grad
        momentum_sgd_delta = v

        param.data += lr * (rmsprop_delta + momentum_sgd_delta)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if eps == 0:
            raise ValueError(
                'eps of RMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))

        # -------- LARS --------
        xp = cuda.get_array_module(param.data)
        if self._lars:
            lr = hp.lr * xp.norm(param.data) / \
                (xp.norm(param.grad) +
                 hp.weight_decay * xp.norm(param.data))
        else:
            lr = hp.lr

        cuda.elementwise(
            'T grad, T lr, T alpha_sgd, T mu1, T alpha_rmsprop, T mu2, T eps',
            'T param, T ms, T v',
            '''
                // -------- RMSProp part --------
                ms = mu2 * ms + (1 - mu2) * grad * grad;
                T rmsprop_delta = -(alpha_rmsprop * grad / (sqrt(ms) + eps));

                // -------- Momentum SGD part --------
                v = mu1 * v - alpha_sgd * grad;
                T momentum_sgd_delta = v;

                param += lr * (rmsprop_delta + momentum_sgd_delta);
            ''',
            'rmsprop_warmup'
        )(grad, lr, hp.alpha_sgd, hp.mu1, hp.alpha_rmsprop, hp.mu2, eps,
          param.data, self.state['ms'], self.state['v'])


class RMSpropWarmup(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate.
        alpha_sgd (float): Learning rate.
        mu1 (float): Exponential decay rate of the first order moment.
        alpha_rmsprop (float): Learning rate.
        mu2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self,
                 lr=_default_hyperparam.lr,
                 alpha_sgd=_default_hyperparam.alpha_sgd,
                 mu1=_default_hyperparam.mu1,
                 alpha_rmsprop=_default_hyperparam.alpha_rmsprop,
                 mu2=_default_hyperparam.mu2,
                 eps=_default_hyperparam.eps,
                 lars=False):
        super(RMSpropWarmup, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha_sgd = alpha_sgd
        self.hyperparam.mu1 = mu1
        self.hyperparam.alpha_rmsprop = alpha_rmsprop
        self.hyperparam.mu2 = mu2
        self.hyperparam.eps = eps
        self._lars = lars

    lr = optimizer.HyperparameterProxy('lr')
    alpha_sgd = optimizer.HyperparameterProxy('alpha_sgd')
    mu1 = optimizer.HyperparameterProxy('mu1')
    alpha_rmsprop = optimizer.HyperparameterProxy('alpha_rmsprop')
    mu2 = optimizer.HyperparameterProxy('mu2')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return RMSpropWarmupRule(self.hyperparam, self._lars)


class RMSpropWarmupScheduler(chainer.training.extension.Extension):
    default_name = 'rmsprop_warmup_scheduler'
    priority = chainer.training.extension.PRIORITY_WRITER

    def __init__(self, n_processes, batchsize, beta_center=10, beta_period=5,
                 optimizer=None):
        self._lr_base = 0.1 * n_processes * batchsize * 0.00390625
        self._beta_center = beta_center
        self._beta_period = beta_period
        self._optimizer = optimizer

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        self._lr = getattr(optimizer, 'lr')
        self._update(trainer)

    def __call__(self, trainer):
        self._update(trainer)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update(self, trainer):
        # CANNOT use ``optimizer.epoch``, MUST use ``trainer.updater.epoch``.
        epoch = trainer.updater.epoch

        if epoch < 40:
            lr = self._lr_base * 0.5
        elif epoch < 70:
            lr = self._lr_base * 0.075
        elif epoch < 85:
            lr = self._lr_base * 0.01
        else:
            lr = self._lr_base * 0.001

        if epoch < self._beta_center:
            alpha_sgd = 0.5 * math.exp(
                2 * (epoch - self._beta_center) / self._beta_period)
        elif epoch < self._beta_center + 0.5 * self._beta_period:
            alpha_sgd = 0.5 + 2 * (epoch - self._beta_center) \
                / self._beta_period
        else:
            alpha_sgd = 1

        alpha_rmsprop = (1 - alpha_sgd) * 0.0003 / lr

        optimizer = self._get_optimizer(trainer)
        setattr(optimizer, 'lr', lr)
        setattr(optimizer, 'alpha_sgd', alpha_sgd)
        setattr(optimizer, 'alpha_rmsprop', alpha_rmsprop)
