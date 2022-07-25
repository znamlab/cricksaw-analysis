#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o cellfinder.out
#SBATCH -e cellfinder.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk
#SBATCH --exclude=gpu[000-036]

root_path="/camp/lab/znamenskiyp/data/instruments/raw_data/projects/"
project="rabies_barcoding"
mouse="BRJN100.4e"
cell_file="$root_path/$project/$mouse/stitchedImages_100/3"
background_file="$root_path/$project/$mouse/stitchedImages_100/2"
model="/camp/lab/znamenskiyp/home/shared/resources/cellfinder_resources/cellfinder_training/sixth_train/model-epoch.30-loss-0.004.h5"
output_dir="/camp/lab/znamenskiyp/home/shared/projects/$project/$mouse/cellfinder_results"

ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate cellfinder
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/

echo "Running cellfinder"
# Just an example. See the user guide for the specific parameters
cellfinder -s $cell_file -b $background_file -o $output_dir -v 25 2 2 --orientation psl  --trained-model $model
