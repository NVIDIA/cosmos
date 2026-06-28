#!/bin/bash
set -x
srun -n1 --ntasks-per-node=1 --partition $1 --gres=gpu:8 --quotatype=reserved --job-name vlmeval --cpus-per-task=64 torchrun --nproc-per-node=8 run.py ${@:2}