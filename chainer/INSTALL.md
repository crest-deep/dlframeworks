# Install Chainer and ChainerMN

To install ChainerMN, you need

- CuPy
- MPI4Py
- Chainer

Following document describes a simple route to install these modules
**from source**, if you prefer using `pip` to download and install a stable
release, check the offical documentation.

For Chainer and ChainerMN, this document use `pip install -e <path>` instead
of `python setup.py install`.
If you are NOT considering to make changes to the source code, you can just
use `python setup.py install`.

Other modules that are not automatically installed, you need to install
manually.

```sh
pip install pillow  # Needed for training ImageNet in Chainer and ChainerMN.
```

We will use following packages:

- CUDA 8.0.61
- cuDNN 7.0
- NCCL 2.1
- Open MPI 2.1.2


## CuPy

We need to add `/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs` to the `LD_LIBRARY_PATH`
and `LIBRARY_PATH` before building CuPy.

### Install

```sh
export LD_LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs:$LIBRARY_PATH
module purge
module load cuda/8.0.61
module load cudnn/7.0
module load nccl/2.1
cd ~/chainer/cupy
git pull
python setup.py clean --all
python setup.py install
```

### Validate

```sh
qrsh -l q_node=1 -l h_rt=00:10:00
# >>> Connect to compute node
module load cuda/8.0.61
module load cudnn/7.0
module load nccl/2.1
python
```

```python
import cupy
import cupy.cudnn
x = cupy.array([1, 2, 3])
```

## MPI4Py

### Install

```sh
module purge
module load cuda/8.0.61
module load openmpi/2.1.2
cd ~/chainer/mpi4py
git pull
python setup.py clean --all
python setup.py install
```

### Validate

```sh
qrsh -l q_node=1 -l h_rt=00:10:00
# >>> Connect to compute node
module load cuda/8.0.61
module load openmpi/2.1.2
python
```

```python
import mpi4py
import mpi4py.MPI
comm = mpi4py.MPI.COMM_WORLD
comm.rank
comm.size
```

## Chainer

### Install

```sh
module purge
module load cuda/8.0.61
module load cudnn/8.0.61
module load nccl/2.1
cd ~/chainer/chainer
git pull
python setup.py clean -all
pip install -e ~/chainer/chainer
```

### Validate

```sh
qrsh -l q_node=1 -l h_rt=00:10:00
# >>> Connect to compute node
module load cuda/8.0.61
module load cudnn/7.0
module load nccl/2.1
python
```

```python
import chainer
```

## ChainerMN

### Install

```sh
module load cuda/8.0.61
module load cudnn/7.0
module load nccl/2.1
module load openmpi/2.1.2
cd ~/chainer/chainermn
git pull
python setup.py clean --all
pip install -e ~/chainer/chainermn
```

### Validate

```sh
qrsh -l q_node=1 -l h_rt=00:10:00
# >>> Connect to compute node
module load cuda/8.0.61
module load cudnn/7.0
module load nccl/2.1
module load openmpi/2.1.2
python
```

```python
import chainermn
import chainermn.nccl
```
