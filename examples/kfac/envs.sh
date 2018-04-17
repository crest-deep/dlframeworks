export PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"
if hash pyenv 2>/dev/null; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
fi

export MODULEPATH=${MODULEPATH}:${HOME}/.modulefiles
if [[ "$(hostname)" = *kfc* ]]; then
  module load cuda/9.0
  module load local/nccl/2.1.15/cuda-9.0
  module load local/cudnn/7.1/cuda-9.0
  module load local/openmpi/2.1.3/cuda-9.0
else
  module load cuda/8.0.61
  module load nccl/local/2.1.15
  module load cudnn/7.0
  module load openmpi/2.1.2-thread-multiple
fi
