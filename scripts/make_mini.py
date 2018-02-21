#!/usr/bin/env python
import argparse
import os
import random


def gen_labels(n):
    labels = []
    for i in range(n):
        r = random.randint(1, 1000) 
        while r in labels:
            r = random.randint(1, 1000)
        labels.append(r)
    return labels

def make_mini(src, dst, labels):
    with open(src, 'r') as src_f, open(dst, 'w') as dst_f:
        for i, line in enumerate(src_f):
            print('{} / {}'.format(i + 1, 1281167))
            label = int(line.split(' ')[1])
            print(label)
            if label in labels:
                dst_f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='/gs/hs0/tgb-crest-deep/data/images/ilsvrc12',
                        help='Path to the directory of full ILSVRC12 dataset,\
                        which must contain train.txt and val.txt file.')
    parser.add_argument('--dst', default='.',
                        help='Path to the directory of output files are\
                        created. The output files are named train{nnn}.txt\
                        and val{nnn}.txt which {nnn} is replaced as a\
                        zero-padded number of classes.')
    parser.add_argument('--num', '-n', type=int, required=True,
                        help='Number of classes that is created.')
    args = parser.parse_args()

    if not (args.num < 1000 and args.num > 0):
        raise ValueError('Invalid number of classes: {}'.format(args.num))

    labels = gen_labels(args.num)

    train_src = os.path.join(args.src, 'train.txt')
    train_dst = os.path.join(args.dst, 'train{:03d}.txt'.format(args.num))
    if os.path.exists(train_dst):
        raise ValueError('File already exists: {}'.format(train_dst))
    print(train_src, 'to', train_dst)
    make_mini(train_src, train_dst, labels)

    val_src = os.path.join(args.src, 'val.txt')
    val_dst = os.path.join(args.dst, 'val{:03d}.txt'.format(args.num))
    if os.path.exists(val_dst):
        raise ValueError('File already exists: {}'.format(val_dst))
    print(val_src, 'to', val_dst)
    make_mini(val_src, val_dst, labels)


if __name__ == '__main__':
    main()
