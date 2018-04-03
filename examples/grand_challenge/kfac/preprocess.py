# ****************************************************************
# Preprocess for training
#
#   This script does three thing
#       1. Generate a job script using the config file.
#       2. Copy the setting filesa and Python source codes to the result dir.
#       3. Prints the generated job script's full path to stdout.
#
# ****************************************************************
import argparse
import datetime
import json
import os
import shutil
import tarfile
import yaml
from socket import gethostname

import preprocess_core


def parse_options(options):
    hostname = gethostname()
    if hostname == 'kfc.r.gsic.titech.ac.jp':
        shell_script = """\
#!/bin/sh
#SBATCH --nodes={nnodes}
#SBATCH --job-name={jobname}
#SBATCH --time={walltime}
#SBATCH --exclude=kfc039
#SBATCH -p k80
#SBATCH -o {result_direcory}/%A.log

""".format(**options)
    else:
        shell_script = """\
#!/bin/sh
#$ -cwd
#$ -l {nodetype}={nnodes}
#$ -l h_rt={walltime}
#$ -N {jobname}
#$ -o {result_direcory}/$JOB_ID.log
#$ -j y

. /etc/profile.d/modules.sh
. ${{HOME}}/.local/modules.sh

""".format(**options)

    for k, v in options['modules'].items():
        shell_script += 'module load {}\n'.format(v)
    
    shell_script += """\
source ./modules.sh

mkdir -p {result_direcory}

cd {working_direcory}

module list

echo ""
echo "---------------- PATH ----------------"
echo $PATH | tr ":" "\\n"
echo "--------------------------------------"
echo ""
echo "---------------- LD_LIBRARY_PATH ----------------"
echo $LD_LIBRARY_PATH | tr ":" "\\n"
echo "-------------------------------------------------"
echo ""
echo "---------------- Python ----------------"
python --version
echo "----------------------------------------"
echo ""
echo "---------------- MPI ----------------"
mpirun --version
echo "-------------------------------------"
echo ""
echo "Job started on $(date)"
echo "................................"

export CUDA_CACHE_DISABLE=1

mpirun \\
  -npernode {npernode} \\
  -np {np} \\
  -output-proctable \\
  -mca pml ob1 \\
  -hostfile hosts \\
  -x PATH \\
  -x LD_LIBRARY_PATH \\
  -x CUDA_CACHE_DISABLE \\
  -x CUDA_HOME \\
  -x CUDA_PATH \\
  -x CUDA_TOP \\
  -x NCCL_IB_CUDA_SUPPORT \\
  -x NCCL_IB_SL \\
  python -W ignore ./main.py \\
    {train} \\
    {val} \\
    --train-root {train_root} \\
    --val-root {val_root} \\
    --arch {arch} \\
    --batchsize {batchsize} \\
    --epoch {epoch} \\
    --loaderjob {loaderjob} \\
    --mean {mean} \\
    --out {result_direcory} \\
    --communicator {communicator} \\
    --loadtype {loadtype} \\
    --iterator {iterator} \\
    --optimizer {optimizer} \\
    --lr {lr} \\
    --momentum {momentum} \\
    --cov_ema_decay {cov_ema_decay} \\
    --inv_freq {inv_freq} \\
    --damping {damping} \\
    --inv_alg {inv_alg} \\
    --cov-batchsize {cov_batchsize} \\
    --n-cov-workers {n_cov_workers} \\
    --n-inv-workers {n_inv_workers} \\
    --npergroup {npergroup} \\
    --comm-core {comm_core} \\
    --nclasses {nclasses} \\
""".format(**options)
    if options['join_cov'] is not None:
        shell_script += '    --join-cov \\\n'.format(**options)
    if options['use_doubly_factored'] is not None:
        shell_script += '    --use_doubly_factored \\\n'.format(**options)
    if options['initmodel'] is not None:
        shell_script += '    --initmodel {initmodel} \\\n'.format(**options)
    if options['resume'] is not None:
        shell_script += '    --resume {resume} \\\n'.format(**options)
    if options['test'] is not None:
        shell_script += '    --test \\\n'

    shell_script += """\


echo "................................"
echo "Job ended on $(date)"
""".format(**options)
    return shell_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    parser.add_argument('-o', '--out', default='main.sh')
    args = parser.parse_args()

    options = preprocess_core.load_options(args.config)
    options['time'] = preprocess_core.get_time()
    options['working_direcory'] = os.path.join(options['out'], options['time'])
    options['result_direcory'] = os.path.join(options['working_direcory'],
                                              'result')
    options['np'] = preprocess_core.get_np(options)
    options['npernode'] = preprocess_core.get_npernode(options)

    shell_script = parse_options(options)

    with open(args.out, 'w') as f:
        f.write(shell_script)

    src_dst = options['working_direcory']
    preprocess_core.copy_code(src_dst)

    log_dst = options['result_direcory']
    os.makedirs(log_dst, exist_ok=True)

    latest_dst = os.path.join(options['out'], 'latest')
    if os.path.exists(latest_dst):
        os.unlink(latest_dst)
    os.symlink(src_dst, latest_dst)

    #print(os.path.join(src_dst, args.out))  # Stdout is passed to `submit` script.
    dirname = os.path.dirname(os.path.join(src_dst, args.out))
    print(dirname)


if __name__ == '__main__':
    main()
