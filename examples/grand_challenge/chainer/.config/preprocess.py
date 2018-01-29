import argparse
import datetime
import json
import os
import shutil
import tarfile
import toml
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
    _, extension = os.path.splitext(filepath)
    if extension == '.json':
        loader = json.load
    elif extension == '.yaml' or extension == '.yml':
        loader = yaml.load
    else:
        raise ValueError('This file type is not supported.')
    with open(filepath, 'r') as f:
        return loader(f)


def get_npernode(options):
    return npernodes[options['nodetype']]


def get_np(options):
    npernode = get_npernode(options)
    return npernode * options['nnodes']


def get_time(format='%y.%m.%d_%H.%M.%S'):
    date = datetime.datetime.now()
    return date.strftime(format)


def parse_options(options):
    options['time'] = get_time()
    shell_script = """\
#!/bin/sh
#$ -cwd
#$ -l {nodetype}={nnodes}
#$ -l h_rt={walltime}
#$ -N {jobname}
#$ -o {out}/{time}/{stdout}
#$ -e {out}/{time}/{stderr}
""".format(**options)
    others = options['others']
    if others is not None:
        for k, v in others.items():
            shell_script += '#$ -{} {}\n'.format(k, v)

    shell_script += """\

source {modules}

mkdir -p {out}/{time}

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

""".format(**options)

    options['np'] = get_np(options)
    options['npernode'] = get_npernode(options)
    if options['train'] is None:
        options['train'] = os.path.join(options['dataset'], 'train.txt')
    if options['val'] is None:
        options['val'] = os.path.join(options['dataset'], 'val.txt')
    if options['mean'] is None:
        options['mean'] = os.path.join(options['dataset'], 'mean.npy')

    shell_script += """\
set -v

mpirun \\
  -output-proctable \\
  -mca pml ob1 \\
  -np {np} \\
  -npernode {npernode} \\
  -x PATH \\
  -x LD_LIBRARY_PATH \\
  python ./train_imagenet.py {train} {val} \\
    --arch {arch} \\
    --epoch {epoch} \\
    --batchsize {batchsize} \\
    --mean {mean} \\
    --out {out}/{time} \\
    --root {root} \\
""".format(**options)
    if options['initmodel'] is not None:
        shell_script += '        --initmodel {initmodel} \\\n'.format(
            **options)
    if options['test'] is not None:
        shell_script += '        --test \\\n'

    shell_script += """\

set +v

echo "................................"
echo "Job ended on $(date)"
""".format(**options)
    return shell_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='config.yaml')
    parser.add_argument('--out', default='train_imagenet.sh')
    args = parser.parse_args()

    options = load_options(args.conf)
    shell_script = parse_options(options)
    with open(args.out, 'w') as f:
        f.write(shell_script)


if __name__ == '__main__':
    main()
