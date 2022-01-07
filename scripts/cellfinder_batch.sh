#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cellfinder.out
#SBATCH -e cellfinder.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@domain.com

cell_file='/path/to/signal/channel'
background_file='path/to/background/channel'
output_dir='/path/to/output/directory'

echo "Loading conda environment"
module load miniconda
conda activate cellfinder

echo "Running cellfinder"
# Just an example. See the user guide for the specific parameters
cellfinder -s $cell_file -b $background_file -o $output_dir -v 5 2 2 --orientation psl
