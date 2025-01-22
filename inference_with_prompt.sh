#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opennmt3

python inference.py \
-model trained_models/retrosnyhesis_ReactSeq_with_prompt_model_on_50k_aug100.pt \
-src datasets/50k_ReactSeq_with_prompt/aug20_test/src_aug20_test.txt \
-output output/tgt_50k_ReactSeq_with_prompt_aug100_train_aug20_test_infer.txt \
-gpu 1 \
-beam_size 10 \
-n_best 10 \
-batch_size 16384 \
-batch_type tokens \
-max_length 500 \
-seed 0 \