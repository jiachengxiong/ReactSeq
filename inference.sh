#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opennmt3

# python inference.py -config config/50k_e_smlies_translate_aug20_test.yml

python inference.py \
-model trained_models/retrosnyhesis_E_smiles_model_on_50k_aug100.pt \
-src datasets/50k_e_smiles/aug20_test/src_aug20_test.txt \
-output output/tgt_50k_e_smiles_aug100_train_aug20_test_infer.txt \
-gpu 7 \
-beam_size 10 \
-n_best 10 \
-batch_size 16384 \
-batch_type tokens \
-max_length 500 \
-seed 0

