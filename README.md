# 実験用コード置き場

実験用コード置き場。

## 実行例

    GPUS=$(nvidia-smi --list-gpus | wc -l)
    export BETTER_EXCEPTIONS=1

    mpirun -np $GPUS python cifar100.py
