#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o brainreg.out
#SBATCH -e brainreg.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk


root_path="/camp/lab/znamenskiyp/data/instruments/raw_data/projects/"
project="hey2_3d-vision_foodres_20220101"
mouse="PZAH6.4b"
subfolder="brainsaw"
red_folder="$root_path/$project/$mouse/$subfolder/stitchedImages_100/3"
green_folder="$root_path/$project/$mouse/$subfolder/stitchedImages_100/2"
atlas="allen_mouse_10um"
output_dir="/camp/lab/znamenskiyp/home/shared/projects/$project/$mouse/brainreg_results"

ml Anaconda3
echo 'Sourcing conda'
source /camp/apps/eb/software/Anaconda/conda.env.sh

echo "Loading conda environment"
conda activate cellfinder
echo "Export library path"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/

echo "Running brainreg"
# Just an example. See the user guide for the specific parameters
brainreg $red_folder $output_dir -a $green_folder -v 8 2 2 --orientation psl  --atlas $atlas
