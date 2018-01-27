import chainer
import numpy
import os
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import random
import six


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


class ImageNetDataset(chainer.dataset.DatasetMixin):

    def __init__(self, pairs, mean, crop_size, root='.', dtype=numpy.float32,
                 label_dtype=numpy.int32, random=True):
        _check_pillow_availability()
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
        self._mean = mean.astype('f')
        self._crop_size = crop_size
        self._root = root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._random = random

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype)
        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]

        image = image.transpose(2, 0, 1)
        label = numpy.array(int_label, dtype=self._label_dtype)
        crop_size = self._crop_size
        _, h, w = image.shape

        if self._random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self._mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
