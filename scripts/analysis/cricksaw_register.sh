#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 128G # memory pool for all cores
#SBATCH -n 10  # maximum number of tasks
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cricksaw_reg.out
#SBATCH -e cricksaw_reg.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk


ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate brainregister
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/

echo "Running script"
python cricksaw_register.py