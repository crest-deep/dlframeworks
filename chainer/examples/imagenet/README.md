# Training ImageNet


## Dataset preparation

To train ImageNet in Chainer (or ChainerMN), you need:

- To crop the RGB images to 256x256.
- To create a `mean.npy` file which is a mean of the whole dataset.
- To create text files that contains `<path to an image> <label number>` on each line (each for training, validating, and testing).
- To install Python `pillow` package.

The original data is located at `/gs/hs0/tgb-crest-deep/data/ilsvrc12`.


## Configure your training

There is a file named `config.yaml`, this file defines the paths and training
settings (e.g. number of nodes, models). You should take a look to this file
before training.

Since the job script is dynamically created from this file, you do not have to
write your own job script. The file `generator.py` does this parsing.


## Submitting to job scheduler

There is a file named `submit`, this invoke `generator.py` and submit the
generated job script (`train.sh`) using `qsub`.
