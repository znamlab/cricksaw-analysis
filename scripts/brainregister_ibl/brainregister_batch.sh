#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 128G # memory pool for all cores
#SBATCH -n 1
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o brainregister.out
#SBATCH -e brainregister.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=blota@crick.ac.uk



project="hey2_3d-vision_foodres_20220101"
mouse="PZAH5.6a"
param_file="brainregister/brainregister_parameters.yaml"
output_dir="/camp/lab/znamenskiyp/home/shared/projects/$project/$mouse/brainregister"

cd $output_dir

ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate brainregister
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/brainregister/lib/

echo "Running brainregister"
# Just an example. See the user guide for the specific parameters
brainregister $param_file
