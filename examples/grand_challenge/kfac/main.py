import argparse
import chainer
from chainer import training
import chainer.cuda
from chainer.dataset import dataset_mixin
from chainer.training import extensions
import chainermn
import multiprocessing
import numpy as np

import dlframeworks

import models_v2.alex as alex
import models_v2.googlenet as googlenet
import models_v2.googlenetbn as googlenetbn
import models_v2.nin as nin
import models_v2.resnet50 as resnet50


class DummyDataset(dataset_mixin.DatasetMixin):

    def __len__(self):
        return np.inf

    def get_example(self, i):
        return np.array([0], dtype=np.float32)


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


def main():
    if not chainer.cuda.available:
        raise RuntimeError("ImageNet requires GPU support.")

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin')
    parser.add_argument('--batchsize', '-B', type=int, default=32)
    parser.add_argument('--epoch', '-E', type=int, default=2)
    parser.add_argument('--initmodel')
    parser.add_argument('--loaderjob', '-j', type=int)
    parser.add_argument('--mean', '-m', default='mean.npy')
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--train-root', default='.')
    parser.add_argument('--val-root', default='.')
    parser.add_argument('--val-batchsize', '-b', type=int, default=250)
    parser.add_argument('--communicator', default='hierarchical')
    parser.add_argument('--loadtype', default='original')
    parser.add_argument('--iterator', default='process')
    parser.add_argument('--optimizer', default='rmsprop_warmup')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    comm = dlframeworks.chainer.communicators.KFACCommunicator(
        args.communicator)
    device = comm.wcomm.intra_rank  # GPU is related with intra rank
    chainer.cuda.get_device(device).use()

    model = archs[args.arch]()
    model.to_gpu()

    if comm.wcomm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.wcomm.mpi_comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if comm.is_grad_worker:
        # Gradient worker
        mean = np.load(args.mean)
        if args.loadtype == 'development':
            if comm.gcomm.rank == 0:
                train = dlframeworks.chainer.datasets.read_pairs(args.train)
                val = dlframeworks.chainer.datasets.read_pairs(args.val)
            else:
                train = None
                val = None
            train = chainermn.scatter_dataset(train, comm.gcomm, shuffle=True)
            val = chainermn.scatter_dataset(val, comm.gcomm)
            train_dataset = dlframeworks.chainer.datasets.CroppingDataset(
                train, args.train_root, mean, model.insize, model.insize)
            val_dataset = dlframeworks.chainer.datasets.CroppingDataset(
                val, args.val_root, mean, model.insize, model.insize)
        else:
            raise NotImplementedError('Invalid loadtype: {}'.format(args.loadtype))
        if args.iterator == 'process':
            multiprocessing.set_start_method('forkserver')
            train_iterator = chainer.iterators.MultiprocessIterator(
                train_dataset, args.batchsize, n_processes=args.loaderjob)
            val_iterator = chainer.iterators.MultiprocessIterator(
                val_dataset, args.val_batchsize, n_processes=args.loaderjob,
                repeat=False)
        elif args.iterator == 'thread':
            train_iterator = chainer.iterators.MultithreadIterator(
                train_dataset, args.batchsize, n_threads=args.loaderjob)
            val_iterator = chainer.iterators.MultithreadIterator(
                val_dataset, args.val_batchsize, n_threads=args.loaderjob,
                repeat=False)
        else:
            train_iterator = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
            val_iterator = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize,
                                                            repeat=False, shuffle=False)
    elif comm.is_cov_worker:
        # Covariance worker
        mean = np.load(args.mean)
        if args.loadtype == 'development':
            if comm.ccomm.rank == 0:
                train = dlframeworks.chainer.datasets.read_pairs(args.train)
                val = dlframeworks.chainer.datasets.read_pairs(args.val)
            else:
                train = None
                val = None
            train = chainermn.scatter_dataset(train, comm.ccomm, shuffle=True)
            val = chainermn.scatter_dataset(val, comm.ccomm)
            train_dataset = dlframeworks.chainer.CroppingDatasetIO(
                train, args.train_root, mean, model.insize, model.insize)
            val_dataset = dlframeworks.chainer.CroppingDatasetIO(
                val, args.val_root, mean, model.insize, model.insize)
        else:
            raise NotImplementedError('Invalid loadtype: {}'.format(args.loadtype))
        if args.iterator == 'process':
            multiprocessing.set_start_method('forkserver')
            train_iterator = chainer.iterators.MultiprocessIterator(
                train_dataset, args.batchsize, n_processes=args.loaderjob)
            val_iterator = chainer.iterators.MultiprocessIterator(
                val_dataset, args.val_batchsize, n_processes=args.loaderjob,
                repeat=False)
        elif args.iterator == 'thread':
            train_iterator = chainer.iterators.MultithreadIterator(
                train_dataset, args.batchsize, n_threads=args.loaderjob)
            val_iterator = chainer.iterators.MultithreadIterator(
                val_dataset, args.val_batchsize, n_threads=args.loaderjob,
                repeat=False)
        else:
            train_iterator = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
            val_iterator = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize,
                                                            repeat=False, shuffle=False)
    else:
        # Inverse worker
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        train_iterator = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
        val_iterator = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize,
                                                        repeat=False, shuffle=False)

    optimizer = dlframeworks.chainer.optimizers.KFAC(comm)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iterator, optimizer,
                                       device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')

    if comm.is_grad_worker:
        # Evaluator
        evaluator = TestModeEvaluator(val_iterator, model, device=device)
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm.gcomm)
        trainer.extend(evaluator, trigger=val_interval)

        # Some display and output extensions are necessary only for one worker.
        # (Otherwise, there would just be repeated outputs.)
        if comm.gcomm.rank == 0:
            trainer.extend(extensions.dump_graph('main/loss'))
            trainer.extend(extensions.LogReport(trigger=log_interval))
            trainer.extend(extensions.observe_lr(), trigger=log_interval)
            trainer.extend(extensions.PrintReport([
                'epoch', 'iteration', 'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy', 'lr'
            ]), trigger=log_interval)
            trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
