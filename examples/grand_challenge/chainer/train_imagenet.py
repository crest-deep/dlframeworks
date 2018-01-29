#!/usr/bin/env python

from __future__ import print_function
import argparse
import json
import os
import random
import shutil
import sys
import tarfile

import numpy as np

import chainer
import chainer.cuda
from chainer import training
from chainer.training import extensions

import chainermn

import dlframeworks


if chainer.__version__.startswith('1.'):
    import models_v1.alex as alex
    import models_v1.googlenet as googlenet
    import models_v1.googlenetbn as googlenetbn
    import models_v1.nin as nin
    import models_v1.resnet50 as resnet50
else:
    import models_v2.alex as alex
    import models_v2.googlenet as googlenet
    import models_v2.googlenetbn as googlenetbn
    import models_v2.nin as nin
    import models_v2.resnet50 as resnet50

# Check Python version if it supports multiprocessing.set_start_method,
# which was introduced in Python 3.4
major, minor, _, _, _ = sys.version_info
if major <= 2 or (major == 3 and minor < 4):
    sys.stderr.write("Error: ImageNet example uses "
                     "chainer.iterators.MultiprocessIterator, "
                     "which works only with Python >= 3.4. \n"
                     "For more details, see "
                     "http://chainermn.readthedocs.io/en/master/"
                     "tutorial/tips_faqs.html#using-multiprocessiterator\n")
    exit(-1)


def save_parameters(trainer, out):
    updater = trainer.updater
    iteraotr = updater.get_iterator('main')
    optimizer = updater.get_optimizer('main')
    batch_size = iteraotr.batch_size
    hyperparam = optimizer.hyperparam.get_dict()
    parameters = {}
    parameters['hyperparam'] = hyperparam
    parameters['batch_size'] = batch_size
    with open(out, 'w') as f:
        json.dump(parameters, f)


def save_code(dirpath, out, compress=True):
    shutil.copytree(dirpath, out)
    if compress:
        with tarfile.open(out + '.tar.gz', 'w:gz') as tar:
            tar.add(out)
        shutil.rmtree(out)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    if not chainer.cuda.available:
        raise RuntimeError("ImageNet requires GPU support.")

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='hierarchical')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    chainer.cuda.get_device(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(args.mean)
    if comm.rank == 0:
        train = PreprocessedDataset(args.train, args.root, mean, model.insize)
        val = PreprocessedDataset(
            args.val, args.root, mean, model.insize, False)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)

    # --------
    # TODO: Avoid using thread
    # Since MultiprocessIterator crashes in TSUBAME 3.0, we will use
    # MultithreadIterator instead. MultithreadIterator uses Python thread to
    # load images. Note that due to Python GIL, MultithreadIterator can be
    # slower than MultiprocessIterator.
    train_iter = chainer.iterators.MultithreadIterator(train, args.batchsize)
    val_iter = chainer.iterators.MultithreadIterator(val, args.val_batchsize,
                                                     repeat=False)

    # We need to change the start method of multiprocessing module if we are
    # using InfiniBand and MultiprocessIterator. This is because processes
    # often crash when calling fork if they are using Infiniband.
    # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
    """
    multiprocessing.set_start_method('forkserver')
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
    """

    # --------
    # TODO: Switch optimzers dynamically
    actual_optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    actual_optimizer = dlframeworks.chainer.optimizers.RMSpropWarmup()

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(actual_optimizer, comm)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    checkpoint_interval = (10, 'iteration') if args.test else (1, 'epoch')
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')

    checkpointer = chainermn.create_multi_node_checkpointer(
        name='imagenet-example', comm=comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=checkpoint_interval)

    # Create a multi node evaluator from an evaluator.
    evaluator = TestModeEvaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    # --------
    # TODO: Integrate RMSpropWarmupScheduler and RMSpropWarmup
    trainer.extend(dlframeworks.chainer.optimizers.RMSpropWarmupScheduler(
        comm.size, args.batchsize))

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.observe_value(
            'alpha_sgd',
            lambda trainer:
                trainer.updater.get_optimizer('main').alpha_sgd))
        trainer.extend(extensions.observe_value(
            'alpha_rmsprop',
            lambda trainer:
                trainer.updater.get_optimizer('main').alpha_rmsprop))
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if comm.rank == 0:
        save_parameters(trainer, os.path.join(args.out, 'params.json'))
        save_code('.', os.path.join(args.out, 'codes'))

    trainer.run()


if __name__ == '__main__':
    main()
