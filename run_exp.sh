#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J normal-L2-norm
#BSUB -m gpu09

DATASET="cifar10"
DATAROOT="../data"
WORKERS="4"
BATCH_SIZE="64"
IMAGE_SIZE="64"
MAX_ITER="300000"
LR="0.00005"
GPU_DEVICE="0"
DITERS="5"
EXPERIMENT=""
MMD_FEATURE_NORMALIZATION="--normalize_mmd_features"
MMD_KERNEL_SIGMA_SCALES="1.0"
# MMD_KERNEL_SIGMA_SCALES="1.0,0.5,0.2,0.1"

set -- \
  --dataset "$DATASET" \
  --dataroot "$DATAROOT" \
  --workers "$WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --max_iter "$MAX_ITER" \
  --lr "$LR" \
  --gpu_device "$GPU_DEVICE" \
  --Diters "$DITERS"

if [ -n "$EXPERIMENT" ]; then
  set -- "$@" --experiment "$EXPERIMENT"
fi

set -- "$@" $MMD_FEATURE_NORMALIZATION
set -- "$@" --mmd_kernel_sigma_scales "$MMD_KERNEL_SIGMA_SCALES"

python -u mmd_gan.py "$@" 2>&1 | tee "run_$(date +%Y%m%d_%H%M%S).out"
