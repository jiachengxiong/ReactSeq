#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opennmt3

python inference.py \
-model trained_models/retrosynthesis_ReactSeq_model_on_50k_aug100.pt \
-src demo_src.txt \
-output demo_tgt.txt \
-gpu 0 \
-beam_size 10 \
-n_best 10 \
-batch_size 16384 \
-batch_type tokens \
-max_length 500 \
-seed 0 \
