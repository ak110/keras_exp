# 実験用コード置き場

実験用コード置き場。

## horovod

```sh
mpirun -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH python cifar100.py
```

