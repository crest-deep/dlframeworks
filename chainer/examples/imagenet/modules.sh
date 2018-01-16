# System
. /etc/profile.d/modules.sh
module load cuda/8.0.61
module load nccl/2.1
module load cudnn/7.0
module load openmpi/2.1.2

# pyenv
export PYENV_ROOT=${HOME}/.pyenv
export PATH=${PYENV_ROOT}/bin:${PATH}
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
