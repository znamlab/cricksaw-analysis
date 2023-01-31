# cricksaw-analysis

Postprocessing and analysis of Cricksaw data

# Installation

## Cellfinder

`cellfinder` install is described there: https://docs.brainglobe.info/cellfinder/installation/cluster-installation/slurm

It has been tested with `cudatoolkit=11.2 cudnn=8.1`

Following the first steps, you might get some library errors:

```
2021-12-19 20:40:30.766514: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64
```

This can be fixed by pointing to the library folder of your cuda environment:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/`

## Cricksa analysis

Easy:

`pip install 'path/to/cricksaw/analysis'`

# Elastix on camp

Running elastix on camp requires a singularity container and elastix binaries
that are both in `/camp/home/blota/home/shared/resources/elastix`. See 
`scripts/analysis/cricksaw-register.sh` for an example of how to use them.