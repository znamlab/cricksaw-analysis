% A small script to run stitchit on a section manually

project = 'rabies_barcoding';
mouse = 'BRYC64.2h';
root = '/camp/lab/znamenskiyp/data/instruments/raw_data/projects';

addpath(genpath('/camp/lab/znamenskiyp/home/users/blota/code/yamlmatlab'));
addpath(genpath('/camp/lab/znamenskiyp/home/users/blota/code/StitchIt/code'));

chdir(fullfile(root, project, mouse))

preProcessTile(-1)
