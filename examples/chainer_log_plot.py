#!/usr/bin/env python
import argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot

from dlframeworks.chainer import Plotter
from dlframeworks.chainer import plot_log
from dlframeworks.chainer.utils import load_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='log')
    parser.add_argument('--out', default='tmp.pdf')
    args = parser.parse_args()

    x = 'epoch'
    y = 'validation/main/accuracy'

    with Plotter(args.out) as (fig, ax):
        plot_log(load_log(args.log), x, y, ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title('Example plot')


if __name__ == '__main__':
    main()
