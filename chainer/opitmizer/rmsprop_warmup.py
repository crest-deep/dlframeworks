import numpy

from chainer.backends import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr_sgd = 0.01
_default_hyperparam.lr_rmsprop = 3e-4
_default_hyperparam.mu1 = 0.9
_default_hyperparam.mu2 = 0.99
_default_hyperparam.eps = 1e-8
_default_hyperparam.beta_center = 10.0
_default_hyperparam.beta_period = 5.0


class RMSpropWarmupRule(optimizer.UpdateRule):

    """Update rule for RMSpropWarmup.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr_sgd (float): Learning rate for SGD.
        lr_rmsprop (float): Learning rate for RMSprop.
        mu1 (float): Exponential decay rate of the first order moment.
        mu2 (float): Coefficient for the moving avarage of the gradient
            second method.
        eps (float): Small value for the numerical stability.
        beta_center (int):
        beta_period (int):

    """

    def __init__(self, parent_hyperparam=None, lr_sgd=None, lr_rmsprop=None,
                 mu1=None, mu2=None, eps=None, beta_center=None,
                 beta_period=None):
        super(RMSpropWarmupRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr_sgd is not None:
            self.hyperparam.lr_sgd = lr_sgd
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

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
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
        t = self.t
        if t < hp.beta_center:
            alpha_sgd = numpy.exp(2 * (t - hp.beta_center) / hp.beta_period)
        elif t < hp.beta_center + hp.beta_period / 2:
            alpha_sgd = 1 / 2 + 2 * (t - hp.beta_center) / hp.beta_period
        else:
            alpha_sgd = 1
        alpha_rmsprop = (1 - alpha_sgd) * hp.lr_rmsprop / hp.lr_sgd

        m *= hp.mu2
        m += (1 - hp.mu2) * grad * grad
        d *= hp.mu1
        d -= grad * (alpha_sgd + alpha_rmsprop/(numpy.sqrt(m) + eps))
        param.data += hp.lr_sgd * d

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
        cuda.elementwise(
            'T grad, T lr_sgd, T lr_rmsprop, T mu1, T m2, T eps, T beta_center,\
             T beta_period, T t',
            'T param, T m, T d',
            '''
                if (t < beta_center) {
                    T alpha_sgd = __expf(2 * (t - beta_center) / beta_period);
                } else if (t < beta_center + beta_period / 2) {
                    T alpha_sgd = 1 / 2 + (t - beta_center) / beta_period;
                } else {
                    T alpha_sgd = 1;
                }
                T alpha_rmsprop = (1 - alpha_sgd) * lr_rmsprop / lr_sgd;

                m = mu2 * m + (1 - mu2) * grad * grad;
                d = mu1 * d - grad *
                    (alpha_sgd + alpha_rmsprop / (sqrt(m) + eps));
                param -= lr_sgd * d;
            ''',
            'rmsprop_warmup')(
                grad, hp.lr_sgd, hp.lr_rmsprop, hp.mu1, hp.mu2, eps,
                hp.beta_center, hp.beta_period, self.t,
                param.data, self.state['m'], self.state['d'])


class RMSpropWarmup(optimizer.GradientMethod):

    """RMSprop optimizer.

    See: T. Tieleman and G. Hinton (2012). Lecture 6.5 - rmsprop, COURSERA:
    Neural Networks for Machine Learning.

    Args:
        lr_sgd (float): Learning rate for SGD.
        lr_rmsprop (float): Learning rate for RMSprop.
        mu1 (float): Exponential decay rate of the first order moment.
        mu2 (float): Coefficient for the moving avarage of the gradient
            second method.
        eps (float): Small value for the numerical stability.
        beta_center (int):
        beta_period (int):

    """

    def __init__(self,
                 lr_sgd=_default_hyperparam.lr_sgd,
                 lr_rmsprop=_default_hyperparam.lr_rmsprop,
                 mu1=_default_hyperparam.mu1, mu2=_default_hyperparam.mu2,
                 eps=_default_hyperparam.eps,
                 beta_center=_default_hyperparam.beta_center,
                 beta_period=_default_hyperparam.beta_period):
        super(RMSpropWarmup, self).__init__()
        self.hyperparam.lr_sgd = lr_sgd
        self.hyperparam.lr_rmsprop = lr_rmsprop
        self.hyperparam.mu1 = mu1
        self.hyperparam.mu2 = mu2
        self.hyperparam.eps = eps
        self.hyperparam.beta_center = beta_center
        self.hyperparam.beta_period = beta_period

    lr_sgd = optimizer.HyperparameterProxy('lr_sgd')
    lr_rmsprop = optimizer.HyperparameterProxy('lr_rmsprop')
    mu1 = optimizer.HyperparameterProxy('mu1')
    mu2 = optimizer.HyperparameterProxy('mu2')
    eps = optimizer.HyperparameterProxy('eps')
    beta_center = optimizer.HyperparameterProxy('beta_center')
    beta_period = optimizer.HyperparameterProxy('beta_period')

    def create_update_rule(self):
        return RMSpropWarmupRule(self.hyperparam)
