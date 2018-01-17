# Training ImageNet

Training ImageNet ILSVRC2012 on TSUBAME 3.0.

There are two type training in this directory.

1. `momentum_sgd`: Default training configuration in ChainerMN.
2. `rmsprop_warmup`: PFN's large batch training configuration. Using RMSprop + momentum SGD with learning rate scheduling (you can also use LARS).

To run the training script, just move to the directory that contains `train.py`
and run `./submit`. The output will be created at `./results/$JOB_ID`.



## Dataset preparation

To train ImageNet in Chainer (or ChainerMN), you need to:

- Crop the RGB images to 256x256.
- Create a `mean.npy` file which is a mean of the whole dataset.
- Create text files that contains `<path to an image> <label number>` on each line (each for training, validating, and testing).
- Install Python `pillow` package.

The original data is located at `/gs/hs0/tgb-crest-deep/data/ilsvrc12`.


## Training configuration

There is a file named `config.yaml`, this file defines the paths and training
settings (e.g. number of nodes, models). You should take a look to this file
before training.

Since the job script is dynamically created from this file, you do not have to
write your own job script. The file `generator.py` does this parsing.


## Job submitting

There is a file named `submit`, this invoke `generator.py` and submit the
generated job script (`train.sh`) using `qsub`.
