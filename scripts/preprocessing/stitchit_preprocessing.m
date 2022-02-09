% add relevant code folder
addpath(genpath('/camp/lab/znamenskiyp/home/users/blota/code/yamlmatlab'));
addpath(genpath('/camp/lab/znamenskiyp/home/users/blota/code/StitchIt/code'));

% cd to relevant directory
cd /camp/lab/znamenskiyp/data/instruments/raw_data/projects/rabies_barcoding/BRYC64.2h

% run preProcessing
preProcessTiles([], 'illumChans', (1:3));
