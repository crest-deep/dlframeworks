import argparse
import yaml


jobscript = """\
#!/bin/sh
#$ -cwd
#$ -l {resource}={node}
#$ -l h_rt={time}
#$ -N imagenet
#$ -j yes
#$ -o $JOB_ID.log

results_dir={out}/$JOB_ID

source {working_dir}/modules.sh  # Load modules, set Python
mkdir -p ${{results_dir}}

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

set -v
mpirun \\
  -output-proctable \\
  -mca pml ob1 \\
  -np {np} \\
  -npernode {npernode} \\
  -x PATH \\
  -x LD_LIBRARY_PATH \\
  python ./train.py \\
    {dataset_dir}/train.txt \\
    {dataset_dir}/val.txt \\
    --arch {arch} \\
    --epoch {epoch} \\
    --batchsize {batchsize} \\
    --mean {dataset_dir}/mean.npy \\
    --out ${{results_dir}} \\
    --initmodel {initmodels}/{arch}.npz \\
    --test
set +v

echo "................................"
echo "Job ended on $(date)"
"""

npernode = {
    'f_node': 4,
    'h_node': 2,
    'q_node': 1,
    's_core': 1,
    'q_core': 1,
    's_gpu': 1,
}


def load_yaml(path):
    with open(path, 'r') as f:
        conf = yaml.load(f)
    return conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='config.yaml')
    parser.add_argument('--out', default='train.sh')
    args = parser.parse_args()

    conf = load_yaml(args.conf)
    conf['npernode'] = npernode[conf['resource']]
    conf['np'] = int(conf['node']) * int(conf['npernode'])
    with open(args.out, 'w') as f:
        f.write(jobscript.format(**conf))


if __name__ == '__main__':
    main()
