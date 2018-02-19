import argparse
import chainer
import chainer.cuda

from src import a01_init
from src import a02_dataset
from src import a03_iterator
from src import a04_optimizer
from src import a05_trainer

import models_v2.alex as alex
import models_v2.googlenet as googlenet
import models_v2.googlenetbn as googlenetbn
import models_v2.nin as nin
import models_v2.resnet50 as resnet50


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

    parser = argparse.ArgumentParser()
    args = \
        a01_init.parse_args(parser, archs)
    comm, device, model = \
        a01_init.initialize(args, archs)
    train_dataset, val_dataset = \
        a02_dataset.get_dataset(args, comm, model)
    train_iterator, val_iterator = \
        a03_iterator.get_iterator(args, train_dataset, val_dataset)
    optimizer = \
        a04_optimizer.get_optimizer(args, comm, model)
    trainer = \
        a05_trainer.get_trainer(args, comm, model, device, train_iterator,
                                val_iterator, optimizer)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
