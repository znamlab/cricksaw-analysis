#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 128G # memory pool for all cores
#SBATCH -n 10  # maximum number of tasks
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cricksaw_reg.out
#SBATCH -e cricksaw_reg.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=blota@crick.ac.uk


ml Anaconda3
ml Singularity
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate brainregister
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/brainregister/lib/

echo "Create elastix aliases"
elastix_folder="/camp/home/blota/home/shared/resources/elastix"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$elastix_folder/lib/
export PATH=$PATH:$elastix_folder/bin/

alias stransformix="singularity run $elastix_folder/elastix_5.0.1.sif transformix"
alias selastix="singularity run $elastix_folder/elastix_5.0.1.sif elastix"

echo "Running script"
python cricksaw_register.py
