#!/bin/bash
#SBATCH -c 5
#SBATCH --gres=gpu:volta:1

source /etc/profile
cd /home/gridsan/sekim
module load anaconda/2022a
eval "$(conda shell.bash hook)"
source .bashrc
conda activate /home/gridsan/sekim/.conda/envs/e3nn

cd magnetic-ordering/run_0310
python data_preprocess_no_mendeleev.py
python magnetic_ordering_draft.py
