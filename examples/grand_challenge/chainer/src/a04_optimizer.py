import chainer
import chainermn

import dlframeworks


def get_optimizer(args, comm, model):
    if args.optimizer == 'kfac':
        optimizer = dlframeworks.chainer.optimizers.KFAC(comm)
    else:
        if args.optimizer == 'momentum_sgd':
            actual_optimizer = chainer.optimizers.MomentumSGD()
        elif args.optimizer == 'adam':
            actual_optimizer = chainer.optimizers.Adam()
        elif args.optimizer == 'rmsprop_warmup':
            actual_optimizer = dlframeworks.chainer.optimizers.RMSpropWarmup()
        else:
            actual_optimizer = chainer.optimizers.RMSprop()
        optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm)
    optimizer.setup(model)
    return optimizer
