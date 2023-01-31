#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH -n 10  # maximum number of tasks
#SBATCH -t 1-0:0 # time (D-HH:MM)
#SBATCH -o stitchit_preprocessing.out
#SBATCH -e stitchit_preprocessing.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=MYUSERNAME@crick.ac.uk

ml matlab
matlab -r ./redo_preprocessing.m
