# Training ImageNet ILSVRC2012 in TSUBAME 3.0

## Python preparation

Following softwares are required.

- CUDA
- cuDNN
- NCCL
- MPI

Following Python packages are required.

- cupy
- chainer
- chainermn
- mpi4py
- pillow (behalf of PIL)
- pyyaml


## Training configuration

To train the ImageNet, there are several steps before starting the actual
Python script. If you run `submit` (e.g. by doing `./submit` at command line)
following process will be done.

1. `preprocess.py` file is invoked from `submit` file.
2. `preprocess.py` file loads `config.yaml` file.
3. `preprocess.py` file generates `main.sh` file.
4. `preprocess.py` file copies the files in the current directory to the `out` (which is specified in the `config.yaml` file).
5. `preprocess.py` file outputs the path to the `main.sh` and passes it the `qsub` command.
6. Starting the actual Python script `main.py`.

There is a file named `config.yaml` and `config.json` under `.config`
directory. These files define the paths and training settings (e.g. number of
epochs, batchsize, number of nodes, and so on), and it is used when a job
script (a `*.sh` file for job system) is generated. Here the default file names
are `config.yaml` and `main.sh`. To change the file name edit `submit` as

```sh
#!/bin/bash
set -eu

jobscript=$(python preprocess.py --conf config.json --out job.sh)
qsub -g tgc-ebdcrest ${jobscript}
```

for `config.json` and `job.sh`.

### Configuration parameters

| key name | value name | description |
|:--------:|:----------:|:-----------:|
| nnodetype | [f_node, h_node, q_node, s_gpu] | Node type in T3 |
| nnodes | Int value larger than 1 | Number of nodes |
| walltime | [hh:mm:ss] (e.g. 00:20:00 for 20 minutes) | Wall time for the job scheduler |
| jobname | any string | Job name for the job scheduler |
| stdout | any string | File name used for writing stdout of the job |
| stderr | any string | File name used for writing stderr of the job |
| others | dictionary | Any key-value paris that is passed to the job script |
| modules | any string | Path to the file that contains `module load` entries |
| nclasses | [8, 16, 32, 1000] | Number of classes of the dataset, less than 1000 means it uses mini-ImageNet |
| mean.npy | any string | Path to the image mean file |
| arch | [alex, googlenet, googlenetbn, nin, resnet50] | DNN architecture |
| epoch | Int value larger than 1 | Number of epochs |
| batchsize | Int value larger than 1 | Number of batchsize per GPU |
| loaderjob | Int value larger than 1 | Number of processes or threads for laoding images |
| out | any string | Output directory for ChainerMN outputs and coping the files |
| resume | any string | Path to the checkpoint model file, if not necessary specify `null` |
| initmodel | any string | Path to the model initialize file, if not necessary specify `null` |
| test | [true, `null`] | Train as a test mode, which evaluate very frequently (takse much long time). |
| communicator | [pure_nccl, hierarchical, two_dimensional, single_node, flat, naive] | Communicator type in ChainerMN |
| loadtype | [original, custom] | Type of loading image to the memory |
| iterator | [serial, thread, process] | Type of iterator |
| optimizer | [momentum_sgd, adam, rmsprop_warmup, rmsprop] | Type of optimizer |
