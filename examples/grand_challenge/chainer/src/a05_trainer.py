import chainer
from chainer import training
from chainer.training import extensions
import chainermn

import dlframeworks


def get_trainer(args, comm, model, device, train_iterator, val_iterator,
                optimizer):
    if args.optimizer == 'kfac':
        if comm.is_inv_worker:
            train_iterator = [None]
            val_iterator = [None]

    updater = training.StandardUpdater(train_iterator, optimizer,
                                       device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')

    # Evaluator
    evaluator = TestModeEvaluator(val_iterator, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    if args.optimizer == 'rmsprop_warmup':
        scheduler = dlframeworks.chainer.optimizers.RMSpropWarmupScheduler(
            comm.size, args.batchsize)
        trainer.extend(scheduler)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    return trainer


def observe_hyperparam(name, trigger):
    """
    >>> trainer.extend(observe_hyperparam('alpha', (1, 'epoch')))
    """
    def observer(trainer):
        return trainer.updater.get_optimizer('main').__dict__[name]
    return extensions.observe_value(name, observer, trigger=trigger)


# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(chainer.training.extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret
