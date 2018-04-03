import os
import numpy as np
from chainer import iterators
import random


def coinflip():
    return random.choice([True, False])
    # return random.randint(0, 1)


#def random_int(hi):
#    random.randint(0, h - crop_size - 1)

def image_process(img):
    crop_size = 224
    _, h, w = img.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    bottom = top + crop_size
    right = left + crop_size
    # img = il.mosaic_pad(img=img, axes=(1, 2))
    # img = rotate_90(img)
    # img = random_rotate(img, axes=(1, 2), max_angle=45)
    # if coinflip(0.3):
        # img = random_blur(img, axes=(1, 2), cnt_pixels=3)
    # img = flip(img)
    # img = random_scroll(img, axes=(1, 2), cnt_pixels=12)
    # img = il.random_scroll(img=img, axes=(1, 2))
    # img = il.random_zoom(img=img, axes=(1, 2), rot=rot)
    # img = il.center_crop(img=img, axes=(1, 2), new_shape=new_shape)
    # img = il.random_blur(img=img, axes=(1, 2), proba=0.2)
    img = img[:, top:bottom, left:right]
    if coinflip():
        img = img[:, :, ::-1]
    return img


def batch_preprocessing(batch):
    for i in range(len(batch)):
        [img, labels] = list(batch[i])
        img = image_process(img=img)
        batch[i] = tuple([img, labels])
    return batch


def test_batch_preprocessing(batch, params, save_count=10):
    global il
    path = os.path.join(params["path_results"], 'image_prepoc_example')
    os.makedirs(path, exist_ok=True)
    for i in range(len(batch)):
        [img, _] = list(batch[i])
        img = image_process(img=img)
        img = np.rollaxis(img, 0, 3)
    return


class MyIterator(iterators.SerialIterator):
    def __init__(self, dataset, batch_size, params=None, repeat=True, shuffle=True):
        super().__init__(dataset, batch_size, repeat, shuffle)
        self._params = params

    def __next__(self):
        batch = super().__next__()
        batch = batch_preprocessing(batch)
        return batch

    next = __next__

