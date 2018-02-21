#!/usr/bin/env python
import cv2
import pathlib
import multiprocessing
import math
import numpy
import os
import sys

target_h = 256
target_w = 256


def task(src_label, dst):
    dst_label = dst.joinpath(src_label.name)
    dst_label.mkdir(exist_ok=True)
    src_names = src_label.glob('*.JPEG')
    for src_name in src_names:
        dst_name = dst_label.joinpath(src_name.name)
        crop(src_name, dst_name)
    print('{} DONE...'.format(src_label.name))


def spawner(src, dst):
    dst.mkdir(exist_ok=True)
    src_labels = list(src.glob('n*'))
    args = [(src_label, dst) for src_label in src_labels]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(task, args)


def spawner_val(src, dst):
    dst.mkdir(exist_ok=True)
    src_names = list(src.glob('*.JPEG'))
    args = [(src_name, dst) for src_name in src_names]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(task_val, args)


def task_val(src_name, dst):
    dst_name = dst.joinpath(src_name.name)
    crop(src_name, dst_name)
    print('{} DONE...'.format(src_name))


def crop(src, dst, target_h=256, target_w=256):
    src = str(src)
    dst = str(dst)
    img = cv2.imread(src)
    h, w, d = img.shape
    new_h = target_h
    new_w = target_w
    if h > w:
        # Scale the height
        new_h = math.ceil((h / w) * target_h)
    else:
        # Scale the width
        new_w = math.ceil((w / h) * target_w)
    # The ratio is same as original image in scaled_img.
    # The size in scaled_img is larger than the original image.
    scaled_img = cv2.resize(img, (new_w, new_h))

    # Dividing by 2 means offset is 2 sides (e.g. left and right).
    h_offset = math.ceil((new_h - target_h) / 2)
    w_offset = math.ceil((new_w - target_w) / 2)
    cropped_img = scaled_img[h_offset:(h_offset + target_h),
                             w_offset:(w_offset + target_w)]
    cv2.imwrite(dst, cropped_img)


def main():
    #spawner(
    #    pathlib.Path('/gs/hs0/tgb-crest-deep/17M30275/datasets/ILSVRC2012/train'),
    #    pathlib.Path('/gs/hs0/tgb-crest-deep/17M30275/datasets/ILSVRC2012-cropped/train'))
    spawner_val(
        pathlib.Path('/gs/hs0/tgb-crest-deep/17M30275/datasets/ILSVRC2012/val'),
        pathlib.Path('/gs/hs0/tgb-crest-deep/17M30275/datasets/ILSVRC2012-cropped/val'))
    #spawner(
    #    pathlib.Path('/gs/hs0/tgb-crest-deep/17M30275/datasets/ILSVRC2012-8-cropped/train'),
    #    pathlib.Path('/tmp/train'))


if __name__ == '__main__':
    main()
