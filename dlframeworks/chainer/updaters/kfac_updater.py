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
            loss_scale=loss_scale)
