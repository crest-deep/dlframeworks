import chainer
import multiprocessing


def get_iterator(args, train_dataset, val_dataset):
    if args.iterator == 'process':
        # We need to change the start method of multiprocessing module if we
        # are # using InfiniBand and MultiprocessIterator. This is because
        # processes # often crash when calling fork if they are using
        # Infiniband.
        # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
        multiprocessing.set_start_method('forkserver')
        train = chainer.iterators.MultiprocessIterator(
            train_dataset, args.batchsize, n_processes=args.loaderjob)
        val = chainer.iterators.MultiprocessIterator(
            val_dataset, args.val_batchsize, n_processes=args.loaderjob,
            repeat=False)
    elif args.iterator == 'thread':
        # Since MultiprocessIterator crashes in TSUBAME 3.0, we will use
        # MultithreadIterator instead. MultithreadIterator uses Python thread
        # to # load images. Note that due to Python GIL, MultithreadIterator
        # can be slower than MultiprocessIterator.
        train = chainer.iterators.MultithreadIterator(
            train_dataset, args.batchsize, n_threads=args.loaderjob)
        val = chainer.iterators.MultithreadIterator(
            val_dataset, args.val_batchsize, n_threads=args.loaderjob,
            repeat=False)
    else:
        train = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
        val = chainer.iterators.SerialIterator(val_dataset, args.val_batchsize,
                                               repeat=False, shuffle=False)
    return train, val
