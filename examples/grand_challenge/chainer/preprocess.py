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


npernodes = {
    'f_node': 4,
    'h_node': 2,
    'q_node': 1,
    's_core': None,
    'q_core': None,
    's_gpu': 1,
}


def load_options(filepath):
    """Load options as a Python dict from a file.

    Only Json and Yaml files are supported.

    Args:
        filepath(str): Path to a file which contains options.

    Returns:
        dict: Loaded dictionary.

    """
    _, extension = os.path.splitext(filepath)
    if extension == '.json':
        loader = json.load
    elif extension == '.yaml' or extension == '.yml':
        loader = yaml.load
    else:
        raise ValueError('Extension {} is not supported.'.format(extension))

    with open(filepath, 'r') as f:
        return loader(f)


class TimeSingleton(object):
    """Singleton for getting a time

    To use single timestamp for all, we use singleton.

    """
    _instance = None
    _date = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if self._date is None:
            self._date = datetime.datetime.now()

    def __call__(self, fmt='%y.%m.%d_%H.%M.%S'):
        return self._date.strftime(fmt)


def get_time(fmt='%y.%m.%d_%H.%M.%S'):
    """Get timestamp. 

    Args:
        fmt(str): Format to parse the datetime.
    """
    t = TimeSingleton()
    return t(fmt)


def get_npernode(options):
    return npernodes[options['nodetype']]


def get_np(options):
    npernode = get_npernode(options)
    return npernode * options['nnodes']


def parse_options(options):
    options['time'] = get_time()
    options['result_direcory'] = os.path.join(options['out'], options['time'])
    options['working_direcory'] = os.path.join(
        options['result_direcory'], 'code')

    shell_script = """\
#!/bin/sh
#$ -cwd
#$ -l {nodetype}={nnodes}
#$ -l h_rt={walltime}
#$ -N {jobname}
#$ -o {result_direcory}/{stdout}
#$ -e {result_direcory}/{stderr}
""".format(**options)
    others = options['others']
    if others is not None:
        for k, v in others.items():
            shell_script += '#$ -{} {}\n'.format(k, v)

    shell_script += '. /etc/profile.d/modules.sh\n'
    for k, v in options['modules'].items():
        shell_script += 'module load {}\n'.format(v)

    shell_script += """\

source {vars}

mkdir -p {result_direcory}

cd {working_direcory}  # We need to change the directory

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
pyenv version
python --version
echo "----------------------------------------"
echo ""
echo "---------------- MPI ----------------"
mpirun --version
echo "-------------------------------------"
echo ""
echo "Job started on $(date)"
echo "................................"

""".format(**options)

    options['np'] = get_np(options)
    options['npernode'] = get_npernode(options)

    data_root = '/gs/hs0/tgb-crest-deep/data/images/ilsvrc12'
    if options['nclasses'] == 1000:
        options['train'] = os.path.join(data_root, 'train.txt')
        options['val'] = os.path.join(data_root, 'val.txt')
    else:
        nclasses = int(options['nclasses'])
        train = 'train{:03d}.txt'.format(nclasses)
        val = 'val{:03d}.txt'.format(nclasses)
        options['train'] = os.path.join(data_root, train)
        options['val'] = os.path.join(data_root, val)
    options['train_root'] = os.path.join(data_root, 'train')
    options['val_root'] = os.path.join(data_root, 'val')

    if 'intel' in options['modules']['mpi']:
        mpirun_cmd = """\
mpiexec.hydra \\
  -ppn {npernode} \\
  -n {np} \\
  -print-rank-map \\
""".format(**options)
    elif 'open' in options['modules']['mpi']:
        mpirun_cmd = """\
mpirun \\
  -npernode {npernode} \\
  -np {np} \\
  -output-proctable \\
  -mca pml ob1 \\
  -x PATH \\
  -x LD_LIBRARY_PATH \\
""".format(**options)
    else:
        raise ValueError('No MPI implementation supported: {}'.format(
            options['modules']['mpi']))

    shell_script += mpirun_cmd

    shell_script += """\
  python ./main.py \\
    {train} \\
    {val} \\
    --arch {arch} \\
    --batchsize {batchsize} \\
    --epoch {epoch} \\
    --loaderjob {loaderjob} \\
    --mean {mean} \\
    --out {result_direcory} \\
    --train-root {train_root} \\
    --val-root {val_root} \\
    --communicator {communicator} \\
    --loadtype {loadtype} \\
    --iterator {iterator} \\
    --optimizer {optimizer} \\
""".format(**options)
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


def copy_code(dst, src='.'):
    ignore = shutil.ignore_patterns(
        'README.md', '.gitignore', '.config')
    shutil.copytree(src, dst, ignore=ignore)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='config.yaml')
    parser.add_argument('--out', default='main.sh')
    args = parser.parse_args()

    options = load_options(args.conf)
    shell_script = parse_options(options)
    with open(args.out, 'w') as f:
        f.write(shell_script)
    dst = os.path.join(options['out'], get_time(), 'code')
    copy_code(dst)
    print(os.path.join(dst, args.out))  # Stdout is passed to `submit` script.


if __name__ == '__main__':
    main()
