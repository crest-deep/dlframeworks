import random
import chainer
from protonn.data.imaging.misc import set_size


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, base, mean, crop_size, random_crop=True):
        self.base = base
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        # image = set_size(image, (3, 256, 256))
        _, h, w = image.shape
        shape_original = image.shape

        if self.random_crop:
            # Randomly crop a region and flip the image
            if h > crop_size + 1:
                top = random.randint(0, h - crop_size - 1)
            else:
                top = 0
            if w > crop_size + 1:
                left = random.randint(0, w - crop_size - 1)
            else:
                left = 0
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        #if image.shape[2] < 224:
        #    print(f"original shape: {shape_original}, label: {label}")
        #try:
        #image -= self.mean[:, top:bottom, left:right]
        #except:
            #print(f"im shape: {shape_original}, mean shape: {self.mean.shape}, top:{top}, bottom:{bottom}, left:{left}, right:{right}")
        #    exit(0)
        #image -= 128
        #image /=  255.0
        return image, label
