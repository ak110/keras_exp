#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=1

python exp.py $*
