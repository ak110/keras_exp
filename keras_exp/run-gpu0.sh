#!/bin/bash
set -eux

export CUDA_VISIBLE_DEVICES=0

python $*
