# cricksaw-analysis

Postprocessing and analysis of Cricksaw data

# Installation

## Cellfinder

`cellfinder` install is described there: https://docs.brainglobe.info/cellfinder/installation/cluster-installation/slurm

Following the first steps, you might get some library errors:

```
2021-12-19 20:40:30.766514: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64
```

This can be fixed by pointing to the library folder of your cuda environment:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cellfinder/lib/`
