#!/usr/bin/env python
import argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot

import dlframeworks.chainer
import dlframeworks.chainer.utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='log')
    parser.add_argument('--out', default='chainer_log_plot.pdf')
    args = parser.parse_args()

    x = 'epoch'
    y = 'validation/main/loss'

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    json_log = dlframeworks.chainer.utils.load_log(args.log)
    dlframeworks.chainer.plot_log(json_log, x, y, ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('Example Plot')
    fig.savefig(args.out)
    matplotlib.pyplot.close(fig)


if __name__ == '__main__':
    main()
