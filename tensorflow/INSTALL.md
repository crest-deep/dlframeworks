# Install TensorFlow

To install TensorFlow **from source**, you need

- Bazel


## Bazel

- Version 0.9.0

### Download and unpack

Download and unpack Bazel's distribution archive from
[release page](https://github.com/bazelbuild/bazel/releases).

We will use version **0.9.0**.

```sh
wget https://github.com/bazelbuild/bazel/releases/download/0.9.0/bazel-0.9.0-dist.zip
unzip -d bazel-0.9.0-dist.zip
```

## Build and install

```sh
module load jdk/1.8.0_144
cd bazel-0.9.0
bash ./compile.sh
```

The output will be `output/bazel`.
Put the file under `$HOME/.local/bazel/0.9.0/bin` and add to the `PATH`.

## TensorFlow

- Commit de4b6bb

You might not be able to build TensorFlow in the login node.
It is recommended to configure, build, and install TensorFlow at compute node.

TensorFlow requires to add `/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs` to the
`LD_LIBRARY_PATH` and `LIBRARY_PATH` before building.
TensorFlow requires to CUPTI libraries and headers.

Install Python packages.

```sh
pip install numpy
pip install dev
```

Load modules.

```sh
export LD_LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/lib64/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/apps/t3/sles12sp2/cuda/8.0.61/extras/CUPTI/lib64:$LIBRARY_PATH
export CPATH=/apps/t3/sles12sp2/cuda/8.0.61/extras/CUPTI/include:$CPATH
module laod jdk/1.8.0_144
module load cuda/8.0.61
module load cudnn/7.0
module load openmpi/2.1.2
```

### Configure

```sh
./configure
```

```
WARNING: Output base '/home/9/17M30275/.cache/bazel/_bazel_17M30275/2188d66641d81211c262071aa8500ef5' is on NFS. This may lead to surprising failures and undetermined behavior.
Extracting Bazel installation...
You have bazel 0.9.0- (@non-git) installed.
Please specify the location of python. [Default is /home/9/17M30275/.pyenv/versions/tensorflow/bin/python]:


Found possible Python library paths:
  /home/9/17M30275/.pyenv/versions/tensorflow/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/9/17M30275/.pyenv/versions/tensorflow/lib/python3.6/site-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]:
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]:
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]:
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: y
VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 8.0


Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /apps/t3/sles12sp2/cuda/8.0.61


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /apps/t3/sles12sp2/cuda/8.0.61]:/apps/t3/sles12sp2/free/cudnn/7.0/cuda/8.0


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.0]6.0


Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]: y
MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```

### Build

```sh
bazel build \
  --config=opt \
  --config=cuda \
  //tensorflow/tools/pip_package:build_pip_package
```

### Install

```sh
bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
```

Output:

```
~/tensorflow/cuda-8.0.61/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles ~/tensorflow/cuda-8.0.61/tensorflow
~/tensorflow/cuda-8.0.61/tensorflow
/scr/1171192.1.all.q/tmp.JzmtXf18p5 ~/tensorflow/cuda-8.0.61/tensorflow
Thu Jan 11 23:03:33 JST 2018 : === Building wheel
warning: no files found matching '*.dll' under directory '*'
warning: no files found matching '*.lib' under directory '*'
warning: no files found matching '*.h' under directory 'tensorflow/include/tensorflow'
warning: no files found matching '*' under directory 'tensorflow/include/Eigen'
warning: no files found matching '*' under directory 'tensorflow/include/external'warning: no files found matching '*.h' under directory 'tensorflow/include/google'
warning: no files found matching '*' under directory 'tensorflow/include/third_party'
warning: no files found matching '*' under directory 'tensorflow/include/unsupported'
~/tensorflow/cuda-8.0.61/tensorflow
Thu Jan 11 23:03:53 JST 2018 : === Output wheel file is in: /home/9/17M30275/tensorflow/cuda-8.0.61/tensorflow/../tensorflow_pkg
```
