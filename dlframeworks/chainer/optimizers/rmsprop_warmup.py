import chainer
import math
import numpy


_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.lr_rmsprop = 3e-4
_default_hyperparam.mu1 = 0.9
_default_hyperparam.mu2 = 0.99
_default_hyperparam.eps = 1e-8
_default_hyperparam.beta_center = 10
_default_hyperparam.beta_period = 5
_default_hyperparam.wd = 0.0005


class RMSpropWarmupRule(chainer.optimizer.UpdateRule):

    """Update rule for RMSpropWarmup.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate for SGD.
        lr_rmsprop (float): Learning rate for RMSprop.
        mu1 (float): Exponential decay rate of the first order moment.
        mu2 (float): Coefficient for the moving avarage of the gradient
            second method.
        eps (float): Small value for the numerical stability.
        beta_center (int):
        beta_period (int):
        wd (float): Weight decay

    """

    def __init__(self, parent_hyperparam=None, lr=None, lr_rmsprop=None,
                 mu1=None, mu2=None, eps=None, beta_center=None,
                 beta_period=None, wd=None, lars=False):
        super(RMSpropWarmupRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if lr_rmsprop is not None:
            self.hyperparam.lr_rmsprop = lr_rmsprop
        if mu1 is not None:
            self.hyperparam.mu1 = mu1
        if mu2 is not None:
            self.hyperparam.mu2 = mu2
        if eps is not None:
            self.hyperparam.eps = eps
        if beta_center is not None:
            self.hyperparam.beta_center = beta_center
        if beta_period is not None:
            self.hyperparam.beta_period = beta_period
        if wd is not None:
            self.hyperparam.wd = wd
        self._lars = lars

    def init_state(self, param):
        xp = chainer.cuda.get_array_module(param.data)
        with chainer.cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['d'] = xp.zeros_like(param.data)

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
        m = self.state['m']
        d = self.state['d']

        if self._lars:
            lr = hp.lr * numpy.norm(param.data) / (numpy.norm(param.grad) + hp.wd * numpy.norm(param.data))
        else:
            lr = hp.lr
        m *= hp.mu2
        m += (1 - hp.mu2) * grad * grad
        d *= hp.mu1
        d -= grad * (hp.alpha_sgd + hp.alpha_rmsprop/(numpy.sqrt(m) + eps))
        param.data += lr * d

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

        if self._lars:
            lr = hp.lr * cupy.norm(param.data) / (cupy.norm(param.grad) + hp.wd * cupy.norm(param.data))
        else:
            lr = hp.lr
        chainer.cuda.elementwise(
            'T grad, T lr, T lr_rmsprop, T mu1, T mu2, T eps, T alpha_sgd,\
             T alpha_rmsprop',
            'T param, T m, T d',
            '''
                m = mu2 * m + (1 - mu2) * grad * grad;
                d = mu1 * d - grad *
                    (alpha_sgd + alpha_rmsprop / (sqrt(m) + eps));
                param += lr * d;
            ''',
            'rmsprop_warmup')(
                grad, lr, hp.lr_rmsprop, hp.mu1, hp.mu2, eps,
                hp.alpha_sgd, hp.alpha_rmsprop,
                param.data, self.state['m'], self.state['d'])


class RMSpropWarmup(chainer.optimizer.GradientMethod):

    """RMSprop warmup optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr (float): Learning rate for SGD.
        lr_rmsprop (float): Learning rate for RMSprop.
        mu1 (float): Exponential decay rate of the first order moment.
        mu2 (float): Coefficient for the moving avarage of the gradient
            second method.
        eps (float): Small value for the numerical stability.
        beta_center (int):
        beta_period (int):
        wd (float): Weight decay.

    """

    def __init__(self,
                 lr=_default_hyperparam.lr,
                 lr_rmsprop=_default_hyperparam.lr_rmsprop,
                 mu1=_default_hyperparam.mu1, mu2=_default_hyperparam.mu2,
                 eps=_default_hyperparam.eps,
                 beta_center=_default_hyperparam.beta_center,
                 beta_period=_default_hyperparam.beta_period,
                 wd=_default_hyperparam.wd, lars=False):
        super(RMSpropWarmup, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.lr_rmsprop = lr_rmsprop
        self.hyperparam.mu1 = mu1
        self.hyperparam.mu2 = mu2
        self.hyperparam.eps = eps
        self.hyperparam.beta_center = beta_center
        self.hyperparam.beta_period = beta_period
        self.hyperparam.wd = wd
        self.hyperparam.alpha_sgd = 0
        self.hyperparam.alpha_rmsprop = 0
        self._lars = lars

    lr = chainer.optimizer.HyperparameterProxy('lr')
    lr_rmsprop = chainer.optimizer.HyperparameterProxy('lr_rmsprop')
    mu1 = chainer.optimizer.HyperparameterProxy('mu1')
    mu2 = chainer.optimizer.HyperparameterProxy('mu2')
    eps = chainer.optimizer.HyperparameterProxy('eps')
    beta_center = chainer.optimizer.HyperparameterProxy('beta_center')
    beta_period = chainer.optimizer.HyperparameterProxy('beta_period')
    wd = chainer.optimizer.HyperparameterProxy('wd')
    alpha_sgd = chainer.optimizer.HyperparameterProxy('alpha_sgd')
    alpha_rmsprop = chainer.optimizer.HyperparameterProxy('alpha_rmsprop')

    def create_update_rule(self):
        return RMSpropWarmupRule(self.hyperparam, self._lars)


class RMSpropWarmupScheduler(chainer.training.extension.Extension):
    default_name = 'rmsprop_warmup_scheduler'
    priority = chainer.training.extension.PRIORITY_WRITER

    def __init__(self, n_processes, batchsize, optimizer=None):
        self._lr_base = 0.1 * n_processes * batchsize * 0.00390625
        self._optimizer = optimizer

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        self._lr = getattr(optimizer, 'lr')
        self._lr_rmsprop = getattr(optimizer, 'lr_rmsprop')
        self._beta_center = getattr(optimizer, 'beta_center')
        self._beta_period = getattr(optimizer, 'beta_period')

        self._update_value(trainer)

    def __call__(self, trainer):
        self._update_value(trainer)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # Cannot use ``optimizer.epoch``, MUST use ``trainer.updater.epoch``.
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

        alpha_rmsprop = (1 - alpha_sgd) * self._lr_rmsprop / lr

        setattr(optimizer, 'lr', lr)
        setattr(optimizer, 'alpha_sgd', alpha_sgd)
        setattr(optimizer, 'alpha_rmsprop', alpha_rmsprop)
