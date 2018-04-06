#!/bin/bash
set -eu

. /etc/profile.d/modules.sh
. ${HOME}/.local/modules.sh

module load cuda/8.0.61
module load nccl/local/2.1.15
module load cudnn/7.0
module load openmpi/2.1.2/cuda-8.0.61/thread

hostfile=hosts

mpirun \
  -np "$(wc -l $hostfile | awk '{print $1}')" \
  -npernode 1 \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x LIBRARY_PATH \
  -hostfile $hostfile \
  pkill -u $USER
