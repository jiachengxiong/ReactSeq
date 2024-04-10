import pdb
import os
os.environ ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda:0"
device_id = 0
import argparse
import torch 
import torch.nn as nn
import onmt
import onmt.model_builder
from onmt.models.model_saver import load_checkpoint
from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
import onmt.opts
import onmt.modules
from onmt.inputters.inputter import IterOnDevice
from onmt.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.train_single import _get_model_opts
from onmt.model_builder import build_model, build_embeddings, build_encoder, build_decoder
from onmt.constants import DefaultTokens, ModelTask
import joblib
from tqdm import tqdm

def _get_parser():
    parser = ArgumentParser(description="train.py")
    train_opts(parser)
    return parser

parser = _get_parser()
opt, unknown = parser.parse_known_args()
opt.data = eval(opt.data)

DATASET = 'all'
RXNTYPE = False 

# Load Ckpt without rxn_type
ckpt_path = "../trained_models/retrosynthesis_ReactSeq_model_on_50k_aug100.pt"
checkpoint = load_checkpoint(ckpt_path)
vocabs = dict_to_vocabs(checkpoint['vocab'])
token2id = vocabs['src'].tokens_to_ids
print("vocabs:\n", vocabs)
print("vocabs_to_dict\n", vocabs_to_dict(vocabs))
print("token2id:\n", token2id)

# Load Model
model_opt = _get_model_opts(opt, checkpoint)
transforms_cls = {}

# Build model
model = build_model(model_opt, opt, vocabs, checkpoint, device_id)
model.load_state_dict(checkpoint, strict = False)
model = model.to(device)

## Dataloader
print("Loading data ...")

# without rxn_type (extract the first fold data)
if DATASET == "all":    
    with open(f"../datasets/50k_ReactSeq/aug100_train/src_aug100_train.txt") as f:
        src_data = f.readlines()[:39946]
    with open(f"../datasets/50k_ReactSeq/aug20_test/src_aug20_test.txt") as f:
        src_data.extend(f.readlines()[:5000])
    with open(f"../datasets/50k_ReactSeq/aug20_val/src_aug20_val.txt") as f:
        src_data.extend(f.readlines()[:4995])
    with open(f"../datasets/50k_ReactSeq/aug100_train/tgt_aug100_train.txt") as f:
        tgt_data = f.readlines()[:39946]
    with open(f"../datasets/50k_ReactSeq/aug20_test/tgt_aug20_test.txt") as f:
        tgt_data.extend(f.readlines()[:5000])
    with open(f"../datasets/50k_ReactSeq/aug20_val/tgt_aug20_val.txt") as f:
        tgt_data.extend(f.readlines()[:4995])
            
print(len(src_data))
src_data = [d[:-1] for d in src_data]
src_list = [d.split(" ") for d in src_data]
tgt_data = [d[:-1] for d in tgt_data]
tgt_list = [d.split(" ") for d in tgt_data]
label_list = [d[0] for d in src_list]

def tokens2ids(src):
    return [token2id[token] for token in src]

print("Formating data ...")

# Transform tokens into ids, then transform list into Tensor [1, seq_length, 1]
src_ids = [tokens2ids(src) for src in src_list]
src_list = [torch.tensor(src_id).unsqueeze(-1).unsqueeze(0).to(device) for src_id in src_ids]
src_len_list = [torch.tensor(src.shape[1]).unsqueeze(0).to(device)  for src in src_list]

tgt_ids = [tokens2ids(tgt) for tgt in tgt_list]
tgt_list = [torch.tensor(tgt_id).unsqueeze(-1).unsqueeze(0).to(device) for tgt_id in tgt_ids]
tgt_len_list = [torch.tensor(tgt.shape[1]).unsqueeze(0).to(device)  for tgt in tgt_list]

enc_outs = []
dec_outs = []

# extract embeddings
model.eval()
with torch.no_grad():
    assert len(src_list) == len(src_len_list)
    for (src, src_len, tgt, tgt_len) in tqdm(zip(src_list, src_len_list, tgt_list, tgt_len_list)):
        enc_out, enc_final_hs, src_len = model.encoder(src, src_len)
        enc_outs.append(enc_out)
        
        #dec_in = tgt[:, :-1, :]
        dec_in = tgt
        model.decoder.init_state(src, enc_out, enc_final_hs)
        dec_out, attns = model.decoder(dec_in, enc_out, src_len=src_len, with_align=False)
        dec_outs.append(dec_out)

joblib.dump(enc_outs, f"enc_outs_{DATASET}_without_rxntype_ReactSeq.pkl")
joblib.dump(dec_outs, f"dec_outs_{DATASET}_without_rxntype_ReactSeq.pkl")

# Usage
# python 1_extract_emebddings.py -config ../config/50k_ReactSeq_finetune.yml