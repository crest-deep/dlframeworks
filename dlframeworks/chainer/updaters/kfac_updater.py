from chainer import training
from chainer.dataset import convert

from dlframeworks.chainer.optimizers import KFAC


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
