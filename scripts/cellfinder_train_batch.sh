#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 10  # maximum number of tasks
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cellfinder.out
#SBATCH -e cellfinder.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk
#SBATCH --nodelise=gpu038  # until they update CUDA on other nodes

training_folder="/camp/lab/znamenskiyp/home/shared/resources/cellfinder_resources/cellfinder_training"

cell_file="$raw_root/$project/$mouse/stitchedImages_050/3"
background_file="$raw_root/$project/$mouse/stitchedImages_050/2"
output_dir="$processed_root/$project/$mouse/cellfinder_results"
model="$processed_root/$project/cellfinder_training/first_train/model.h5"

ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate cellfinder
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/

echo "Running cellfinder"
# Just an example. See the user guide for the specific parameters
cellfinder -s $cell_file -b $background_file -o $output_dir -v 8 2 2 --orientation psl --trained-model $model
