#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J normal-L2-norm
#BSUB -m gpu09
python -u mmd_gan.py 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).out
  
