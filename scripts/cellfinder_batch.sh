#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 10  # maximum number of tasks
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cellfinder.out
#SBATCH -e cellfinder.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk
#SBATCH --nodelise=gpu038  # until they update CUDA on other nodes

raw_root="/camp/lab/znamenskiyp/data/instruments/raw_data/projects/"
processed_root="/camp/lab/znamenskiyp/home/shared/projects"
project="rabies_barcoding"
mouse="BRYC64.2h"

cell_file="$raw_root/$project/$mouse/stitchedImages_100/3"
background_file="$raw_root/$project/$mouse/stitchedImages_100/2"
output_dir="$processed_root/$project/$mouse/cellfinder_results"

ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate cellfinder
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/

echo "Running cellfinder"
# Just an example. See the user guide for the specific parameters
cellfinder -s $cell_file -b $background_file -o $output_dir -v 8 1 1 --orientation psl
