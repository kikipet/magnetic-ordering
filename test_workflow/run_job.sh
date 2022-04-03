#!/bin/bash
#SBATCH -c 5
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:volta:1

source /etc/profile
cd /home/gridsan/sekim
module load anaconda/2022a
eval "$(conda shell.bash hook)"
source .bashrc
conda activate /home/gridsan/sekim/.conda/envs/e3nn

cd magnetic-ordering/test_workflow
python magnetic_ordering.py
