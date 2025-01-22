#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opennmt3
python train.py -config config/50k_ReactSeq_finetune_with_prompt.yml