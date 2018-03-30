import chainer
import chainermn


def parse_args(parser, archs):
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
    # parser.add_argument('--optimizer', default='rmsprop_warmup')
    parser.add_argument('--optimizer', default='momentum_sgd')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    return args


def initialize(args, archs):
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank
    chainer.cuda.get_device(device).use()
    model = archs[args.arch]()
    model.to_gpu()
    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    return comm, device, model
