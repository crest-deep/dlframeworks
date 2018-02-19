import chainer
import chainermn
import math
import numpy as np
import os
from PIL import Image
import random
import six
import time


def get_dataset(args, comm, model):
    mean = np.load(args.mean)

    if args.loadtype == 'original':
        if comm.rank == 0:
            # Original will load each image when it is needed. Not load the
            # whole dataset in-memery at training started.
            train = _Dataset(args.train, args.train_root, mean, model.insize)
            val = _Dataset(args.val, args.val_root, mean, model.insize, False)
        else:
            train = None
            val = None

        # This will scatter only the <location, label> pair with the comm
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        val = chainermn.scatter_dataset(val, comm)
    else:
        if comm.rank == 0:
            # Custom will load the whole dataset in the main memory at the
            # start time.
            train = read_locations(args.train)
            val = read_locations(args.val)
        else:
            train = None
            val = None

        # This will scatter only the <location, label> pair with the comm
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        val = chainermn.scatter_dataset(val, comm)
        train = _CustomDataset(train, args.train_root, mean, model.insize)
        val = _CustomDataset(val, args.val_root, mean, model.insize, False)

    return train, val


def read_locations(path):
    locations = []
    with open(path) as f:
        for line in f:
            path, label = line.split()
            label = int(label)
            locations.append((path, label))
    return locations


class _Dataset(chainer.dataset.DatasetMixin):
    """Dataset for ImageNet dataset

    In `PreprocessDataset`  which is defind in the original train_imagenet.py
    script provided from ChainerMN, there is a `base` attribute that reads the
    image and the label from the `*.txt` file. However, this class only
    supports pre-cropped images, we need to change the whole Dataset
    definition to handle raw ImageNet datasets.

    This class is similar to `LabeledImageDataset`, and the difference between
    that this class do:
        1) resizes the image to a larger image than the cropped one
        2) removes alpha dimension (e.g. transparency) if exists
        3) adds a new dimension if grayscale
        4) cropes the image
        5) subtract the mean image from the image 
    in `get_example()` method.

    Args:
        pairs (str or list or tuples): Same as `LabeledImageDataset`.
        root (str): Same as `LabeledImageDataset`
        mean (np.ndarray): Mean image.
        crop_size (int): Cropping size. Height and width will be the same.
        random (bool): Do randam cropping and flopping or not.
        image_dtype: Data type of resulting image arrays.
        label_dtype: Data type of resulting label arrays.

    """
    def __init__(self, pairs, root, mean, crop_size, random=True,
                 image_dtype=np.float32, label_dtype=np.int32):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._mean = mean.astype('f')
        self._crop_size = crop_size
        self._random = random
        self._image_dtype = image_dtype
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, label = self._pairs[i]
        path = os.path.join(self._root, path)
        image = _read_image(path, self._crop_size, self._crop_size, self._mean,
                            self._random, self._image_dtype)
        label = np.array(label, dtype=self._label_dtype)
        return image, label


class _CustomDataset(chainer.dataset.DatasetMixin):

    def __init__(self, locations, root, mean, crop_size, random=True):
        self._locations = locations
        self._root = root
        self._mean = mean
        self._crop_size = crop_size
        self._random = random
        self.prepare()

    def __len__(self):
        return len(self._labels)

    def prepare(self):
        s_time = time.time()
        print("Start loading images...")
        images = []
        labels = []
        n = len(self._locations)
        for i, location in enumerate(self._locations):
            path = os.path.join(self._root, location[0])
            image = _read_image(path, self._crop_size, self._crop_size,
                                self._mean, self._random)
            images.append(image)
            labels.append(location[1])
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        self._images = images
        self._labels = labels
        print('Loading images done.')
        print('Took {} [sec] for {} images'.format(time.time() - s_time, n))

    def get_example(self, i):
        return self._images[i], self._labels[i]


def _read_image(path, crop_h, crop_w, mean, rand, dtype=np.float32):
    """Read an image from path.

    Args:
        path (str): Path to the image file.
        crop_h (int): Cropping height.
        crop_w (int): Cropping width.
        mean (np.ndarray): Mean image file.
        rand (bool): Do randam cropping and flopping or not.
        dtype: Data type of resulting image arrays.
    """
    # Open an image using PIL
    im = Image.open(path)
    w, h = im.size
    # Scale the image to make it larger than the cropped size
    if w < crop_w or h < crop_h:
        if w < h:
            h = math.ceil(h * crop_w / w)
            w = crop_w
        else:
            w = math.ceil(w * crop_h / h)
            h = crop_h
        # We resize here since we cannot resize the image easily in NumPy
        # format.
        im = im.resize((w, h))
    try:
        # Convert image file to NumPy ndarray
        image = np.asarray(im, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(im, 'close'):
            im.close()
        # We do not use actual image file anymore
        del im

    if image.ndim == 2:
        # image is grayscale
        image = image[:, :, np.newaxis]
    image = image[:, :, :3]  # Remove alpha (i.e. transparency)
    image = image.transpose(2, 0, 1)  # (c, h, w) -> (h, w, c)

    if rand:
        # Randomly crop a region and flip the image
        top = random.randint(0, max(h - crop_h - 1, 0))
        left = random.randint(0, max(w - crop_w - 1, 0))
        if random.randint(0, 1):
            image = image[:, :, ::-1]
    else:
        # Crop the center
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    bottom = top + crop_h
    right = left + crop_w

    image = image[:, top:bottom, left:right]
    # image -= mean[:, :h, :w] will cause error in NumPy
    image = image - mean[:, :crop_h, :crop_w]  # Only use top left of mean
    image *= (1.0 / 255.0)  # Scale to [0, 1]
    return image
