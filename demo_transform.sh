#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rdkit2019

python transform.py \
    -src demo_src.txt \
    -tgt demo_tgt.txt \
    -output demo_output_smiles.txt