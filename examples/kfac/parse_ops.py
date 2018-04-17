import argparse
from socket import gethostname
from os import makedirs
from os import unlink
from os import symlink
from os.path import join
from os.path import exists

from parse_ops_core import load_ops
from parse_ops_core import get_time
from parse_ops_core import get_npernode
from parse_ops_core import get_np
from parse_ops_core import copy_files


def parse_ops(ops):
    hostname = gethostname().strip()
    if hostname == 'kfc':
        script = """\
#!/bin/sh
#SBATCH --nodes={nnodes}
#SBATCH --job-name={jobname}
#SBATCH --time={walltime}
#SBATCH --exclude=kfc039
#SBATCH -p k80
#SBATCH -o {result_directory}/%A.log

""".format(**ops)
    else:
        script = """\
#!/bin/sh
#$ -cwd
#$ -l {nodetype}={nnodes}
#$ -l h_rt={walltime}
#$ -N {jobname}
#$ -o {result_directory}/$JOB_ID.log
#$ -j y

. /etc/profile.d/modules.sh
""".format(**ops)

    script += """\
. {envs_file}

mkdir -p {result_directory}
cd {working_directory}


# ======== Show Settings ========
if command -v module >/dev/null 2>&1; then
  module list
fi
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


# ======== Run ========
mpirun \\
  -npernode {npernode} \\
  -np {np} \\
""".format(**ops)

    for mpirun_op in ops['mpirun_ops']:
        script += '  {} \\\n'.format(mpirun_op)

    script += """\
  python -W ignore {working_directory}/main.py \\
    {train} \\
    {val} \\
    --train-root {train_root} \\
    --val-root {val_root} \\
    --arch {arch} \\
    --batchsize {batchsize} \\
    --epoch {epoch} \\
    --loaderjob {loaderjob} \\
    --mean {mean} \\
    --out {result_directory} \\
    --communicator {communicator} \\
    --loadtype {loadtype} \\
    --iterator {iterator} \\
    --lr {lr} \\
    --momentum {momentum} \\
    --cov_ema_decay {cov_ema_decay} \\
    --inv_freq {inv_freq} \\
    --damping {damping} \\
    --cov-batchsize {cov_batchsize} \\
    --n-cov-workers {n_cov_workers} \\
    --n-inv-workers {n_inv_workers} \\
    --npergroup {npergroup} \\
    --weight-decay {weight_decay} \\
""".format(**ops)
    if ops['join_cov'] is not None:
        script += '    --join-cov \\\n'.format(**ops)
    if ops['initmodel'] is not None:
        script += '    --initmodel {initmodel} \\\n'.format(**ops)
    if ops['resume'] is not None:
        script += '    --resume {resume} \\\n'.format(**ops)
    if ops['test'] is not None:
        script += '    --test \\\n'

    script += """\


echo "................................"
echo "Job ended on $(date)"
""".format(**ops)

    return script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', required=True)
    parser.add_argument('-o', '--out', default='main.sh')
    args = parser.parse_args()

    ops = {}
    for config in args.config:
        ops.update(load_ops(config))
    ops['time'] = get_time()
    ops['working_directory'] = join(ops['out'], ops['time'])
    ops['result_directory'] = join(ops['working_directory'], 'result')
    ops['np'] = get_np(ops)
    ops['npernode'] = get_npernode(ops)

    script = parse_ops(ops)

    with open(args.out, 'w') as f:
        f.write(script)

    working_directory = ops['working_directory']
    result_directory = ops['result_directory']
    copy_files(working_directory, *ops['ignored_files'], src='.')
    makedirs(result_directory, exist_ok=True)

    latest = join(ops['out'], 'latest')
    if exists(latest):
        unlink(latest)
    symlink(result_directory, latest)

    print(join(working_directory, args.out))


if __name__ == '__main__':
    main()
