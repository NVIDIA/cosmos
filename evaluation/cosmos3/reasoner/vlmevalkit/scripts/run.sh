#!/bin/bash
set -x
export GPU=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc-per-node=$GPU run.py ${@:1}