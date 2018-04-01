import argparse
import chainer
from chainer import training
import chainer.cuda
from chainer.dataset import dataset_mixin
from chainer.training import extensions
import chainermn
import multiprocessing
import cupy as cp
import numpy as np
import sys

import dlframeworks

import models_v2.alex as alex
import models_v2.googlenet as googlenet
import models_v2.googlenetbn as googlenetbn
import models_v2.nin as nin
import models_v2.resnet50 as resnet50


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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cov_ema_decay', type=float, default=0.99)
    parser.add_argument('--inv_freq', type=int, default=20)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--inv_alg')
    parser.add_argument('--use_doubly_factored', action='store_true')
    parser.add_argument('--cov-batchsize', type=int, default=16)
    parser.set_defaults(test=False)
    args = parser.parse_args()

    comm = dlframeworks.chainer.communicators.KFACCommunicator(
        args.communicator, debug=True, timeout=300, check_value=True)
    device = comm.wcomm.intra_rank  # GPU is related with intra rank
    chainer.cuda.get_device_from_id(device).use()
    model = archs[args.arch]()

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Initialize weights
    x = np.zeros((1, 3, model.insize, model.insize), dtype=np.float32)
    t = np.zeros((1,), dtype=np.int32)
    model(x, t)

    try:
        model.to_gpu()
    except chainer.cuda.cupy.cuda.runtime.CUDARuntimeError as e:
        print('Error occured in {}'.format(comm.wcomm.rank), file=sys.stderr)
        raise e

    if comm.wcomm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.wcomm.mpi_comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    # ======== Create optimizer ========
    optimizer = dlframeworks.chainer.optimizers.KFAC(
        comm,
        lr=args.lr,
        momentum=args.momentum,
        cov_ema_decay=args.cov_ema_decay,
        inv_freq=args.inv_freq,
        damping=args.damping,
        inv_alg=args.inv_alg,
        use_doubly_factored=args.use_doubly_factored,
    )
    # damping ~ 0.035 is good
    optimizer.setup(model)


    if comm.is_grad_worker or comm.is_cov_worker:
        if comm.is_grad_worker:
            # Gradient worker
            # Load all dataset in memory
            dataset_class = dlframeworks.chainer.datasets.CroppingDataset
            sub_comm = comm.gcomm
            batchsize = args.batchsize
        else:
            # Covariance worker
            # Load dataset in memory when needed
            dataset_class = dlframeworks.chainer.datasets.CroppingDatasetIO
            sub_comm = comm.ccomm
            batchsize = args.cov_batchsize

        mean = np.load(args.mean)

        # ======== Create dataset ========
        if comm.gcomm.rank == 0 or comm.ccomm.rank == 0:
            train = dlframeworks.chainer.datasets.read_pairs(args.train)
            val = dlframeworks.chainer.datasets.read_pairs(args.val)
        else:
            train = None
            val = None
        train = chainermn.scatter_dataset(train, sub_comm, shuffle=True)
        val = chainermn.scatter_dataset(val, sub_comm)
        train_dataset = dataset_class(
            train, args.train_root, mean, model.insize, model.insize)
        val_dataset = dataset_class(
            val, args.val_root, mean, model.insize, model.insize)

        # ======== Create iterator ========
        if args.iterator == 'process':
            multiprocessing.set_start_method('forkserver')
            train_iterator = chainer.iterators.MultiprocessIterator(
                train_dataset, batchsize, n_processes=args.loaderjob)
            val_iterator = chainer.iterators.MultiprocessIterator(
                val_dataset, args.val_batchsize, n_processes=args.loaderjob,
                repeat=False)
        elif args.iterator == 'thread':
            train_iterator = chainer.iterators.MultithreadIterator(
                train_dataset, batchsize, n_threads=args.loaderjob)
            val_iterator = chainer.iterators.MultithreadIterator(
                val_dataset, args.val_batchsize, n_threads=args.loaderjob,
                repeat=False)
        else:
            train_iterator = chainer.iterators.SerialIterator(train_dataset, batchsize)
            val_iterator = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize,
                                                            repeat=False, shuffle=False)

        # ======== Create updater ========
        updater = training.StandardUpdater(train_iterator, optimizer,
                                           device=device)

        # ======== Create trainer ========
        if comm.is_cov_worker:
            def stop_trigger(x):
                if x.updater.get_optimizer('main').is_done:
                    return True
                else:
                    return False
            trainer = training.Trainer(updater, stop_trigger, args.out)
        else:
            trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

        # ======== Extend trainer ========
        val_interval = (10, 'iteration') if args.test else (1, 'epoch')
        log_interval = (10, 'iteration') if args.test else (1, 'epoch')
        if comm.is_grad_worker:
            # Only gradient worker needs to join this extension
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
        print('done', comm.wcomm.rank)

    else:
        # Inverse worker
        # ======== Create optimizer ========
        while True:
            optimizer.update()
            if optimizer.is_done:
                break
        print('Inverse done')


if __name__ == '__main__':
    main()
