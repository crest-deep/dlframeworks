import chainer
import math
import numpy as np
import os
from PIL import Image
import random
import six
import time


def read_pairs(path):
    """Read path to image and label pairs from file.

    Args:
        path (str): Path to the image-label pair file.

    Returns:
        list of pairs: Each pair type is ``(str, int)``. Which first element is
            a path to image and second element is a label.
    """
    pairs = []
    with open(path) as f:
        for line in f:
            path, label = line.split()
            label = int(label)
            pairs.append((path, label))
    return pairs


class CroppingDataset(chainer.dataset.DatasetMixin):
    """Dataset for labeled image dataset.

    In ``PreprocessDataset``  which is defind in the original train_imagenet.py
    script provided from ChainerMN, there is a ``base`` attribute that reads
    the path to image and the label from the ``*.txt`` file. However, this
    class only supports pre-cropped images (i.e.. 256 x 256 or larger), so to
    handle the raw ImageNet datasets we implemented a new type of dataset.

    This class is similar to ``LabeledImageDataset`` in Chainer. The difference
    is that this class reads whole dataset in-memory before training starts.

    Before training this class does 2 things.
        1) Rescale the image to make it larger than 244 x 244.
        2) Load every image to the memory.
    This is done by ``_prepare()`` (precisely ``_read_image()``) function.
    After training started, at each time ``get_example()`` is called, this
    class does 3 things.
        1) Add new axis if grayscale, remove one channle if number of channles
            is larger than 3.
        2) Crop the image to 244 x 244.
        3) Subtract mean image from the image.

    To use this class you need pass pairs generated by ``read_pairs()``.

    Args:
        pairs (list): A list of pairs, the ``i``-th element represents a pair
            of the path to the ``i``-th image and the corresponding label.
        root (str): Root directory to retrieve images from. Must be 3 dim,
            ``(c, h, w)``.
        mean (``numpy.ndarray``): Mean image file of the dataset.
        crop_h (int): Cropping height.
        crop_w (int): Cropping width.
        image_dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.

    """

    def __init__(self, pairs, root, mean, crop_h, crop_w, random=True,
                 image_dtype=np.float32, label_dtype=np.int32):
        self._pairs = pairs
        self._root = root
        self._mean = mean.astype('f')
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._random = random
        self._image_dtype = image_dtype
        self._label_dtype = label_dtype
        self._prepare()

    def __len__(self):
        return len(self._pairs)

    def _prepare(self):
        s_time = time.time()
        images = []
        labels = []
        for (path, label) in self._pairs:
            path = os.path.join(self._root, path)
            images.append(_read_image(path, self.crop_h, self.crop_w,
                                      self._image_dtype))
            labels.append(label)
        labels = np.array(labels, dtype=self._label_dtype)
        self._images = images
        self._labels = labels
        e_time = time.time()
        print('took {} sec for {} images'.format(e_time - s_time, len(self)))

    def get_example(self, i):
        crop_h = self._crop_h
        crop_w = self._crop_w
        mean = self._mean
        image = self._images[i]
        if image.ndim == 2:
            # image is grayscale
            image = image[:, :, np.newaxis]
        image = image[:, :, :3]  # Remove alpha (i.e. transparency)
        image = image.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)
        _, h, w = image.shape

        if self._random:
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
        image = image - mean[:, :crop_h, :crop_w]  # Only use top left of mean
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, self._labels[i]


class CroppingDatasetIO(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, root, mean, crop_h, crop_w, random=True,
                 image_dtype=np.float32, label_dtype=np.int32):
        self._pairs = pairs
        self._root = root
        self._mean = mean.astype('f')
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._random = random
        self._image_dtype = image_dtype
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        crop_h = self._crop_h
        crop_w = self._crop_w
        path, label = self._pairs[i]
        image = _read_image(path, crop_h, crop_w, self._image_dtype)
        mean = self._mean
        if image.ndim == 2:
            # image is grayscale
            image = image[:, :, np.newaxis]
        image = image[:, :, :3]  # Remove alpha (i.e. transparency)
        image = image.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)
        _, h, w = image.shape

        if self._random:
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
        image = image - mean[:, :crop_h, :crop_w]  # Only use top left of mean
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, np.array(label, dtype=self._label_dtype)


def _read_image(path, crop_h, crop_w, dtype):
    im = Image.open(path)

    w, h = im.size
    if w < crop_w or h < crop_h:
        if w < h:
            h = math.ceil(h * crop_w / w)
            w = crop_w
        else:
            w = math.ceil(w * crop_h / h)
            h = crop_h
        im = im.resize((w, h))
    try:
        image = np.asarray(im, dtype=dtype)
    finally:
        im.close()
        del im
    return image
