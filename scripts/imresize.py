import argparse
import math
import numpy as np
import os
from PIL import Image
from skimage import io
from skimage.transform import resize
from skimage.transform import rescale
import time
import tqdm


def imresize_PIL(path):
    s_time = time.time()
    im = Image.open(path)
    e_time = time.time()
    open_time = e_time - s_time

    w, h = im.size
    if w < 256 or h < 256:
        if w < h:
            h = math.ceil(h * 256 / w)
            w = 256
        else:
            w = math.ceil(w * 256 / h)
            h = 256
        s_time = time.time()
        im = im.resize((w, h))
        e_time = time.time()
        resize_time = e_time - s_time
    else:
        resize_time = 0

    s_time = time.time()
    try:
        im_array = np.asarray(im, dtype=np.float32)
    finally:
        im.close()
        del im
    e_time = time.time()
    convert_time = e_time - s_time

    return im_array, open_time, resize_time, convert_time


def imresize_skimage(path):
    s_time = time.time()
    im = io.imread(path)
    e_time = time.time()
    open_time = e_time - s_time

    if im.ndim == 2:
        w, h = im.shape
    else:
        w, h, _ = im.shape
    if w < 256 or h < 256:
        if w < h:
            h = math.ceil(h * 256 / w)
            w = 256
        else:
            w = math.ceil(w * 256 / h)
            h = 256
        s_time = time.time()
        im = resize(im, (w, h), mode='reflect')
        e_time = time.time()
        resize_time = e_time - s_time
    else:
        resize_time = 0

    s_time = time.time()
    im_array = np.asarray(im, dtype=np.float32)
    e_time = time.time()
    convert_time = e_time - s_time

    return im_array, open_time, resize_time, convert_time


def imrescale_skimage(path):
    s_time = time.time()
    im = io.imread(path)
    e_time = time.time()
    open_time = e_time - s_time

    if im.ndim == 2:
        w, h = im.shape
    else:
        w, h, _ = im.shape
    if w < 256 or h < 256:
        if w < h:
            scale = math.ceil(h * 256 / w)
        else:
            scale = math.ceil(w * 256 / h)
        s_time = time.time()
        im = rescale(im, scale, mode='reflect')
        e_time = time.time()
        resize_time = e_time - s_time
    else:
        resize_time = 0

    s_time = time.time()
    im_array = np.asarray(im, dtype=np.float32)
    e_time = time.time()
    convert_time = e_time - s_time

    return im_array, open_time, resize_time, convert_time


def measure(func, locations, root):
    s_time = time.time()
    n = len(locations)
    print("Start loading images...")
    open_times = []
    resize_times = []
    convert_times = []
    for location in tqdm.tqdm(locations):
        path = os.path.join(root, location[0])
        im_array, open_time, resize_time, convert_time = func(path)
        open_times.append(open_time)
        resize_times.append(resize_time)
        convert_times.append(convert_time)
    e_time = time.time()
    print("Loading {} images DONE.".format(n))
    print('    total:       {:4f} [sec]'.format(e_time - s_time))
    print('    sum open:    {:4f} [sec]'.format(sum(open_times)))
    print('    sum resize:  {:4f} [sec]'.format(sum(resize_times)))
    print('    sum convert: {:4f} [sec]'.format(sum(convert_times)))


def read_locations(path):
    locations = []
    with open(path) as f:
        for line in f:
            path, label = line.split()
            label = int(label)
            locations.append((path, label))
    return locations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()

    locations = read_locations(args.path)
    print('======== PIL ========')
    measure(imresize_PIL, locations, args.root)
    print()
    print('======== scikit-image ========')
    measure(imresize_skimage, locations, args.root)
    print()
    print('======== scikit-image ========')
    measure(imresize_skimage, locations, args.root)


if __name__ == '__main__':
    main()
