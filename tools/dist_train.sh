#!/bin/bash
GPUS=1

# RANK=0
NODE_COUNT=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
# echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"
echo "node list: ${SLURM_NODELIST}"

export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$NODE_COUNT
export MASTER_ADDR=$MASTER_ADDR

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:2}
# Any arguments from the third one are captured by ${@:2}
