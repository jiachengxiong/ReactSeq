import argparse
import os
import pandas as pd
import random
from rdkit import Chem
from tqdm import tqdm
from tqdm.contrib import tzip
from e_smiles import get_e_smiles, merge_smiles, get_edit_from_e_smiles, merge_smiles_only, get_e_smiles_with_check
from tqdm.contrib.concurrent import process_map  

def preprocess_data(df, data_source, split, augtime):
    """
    Clean few data, and Get ReactSeq,
    """
    if data_source == '50k':
        if split == "train":
            rxn_class_list = []
            e_smiles_list = []
            for class_, rxn in tzip(df['class'], df['reactants>reagents>production']):
                try:
                    o_reactant = rxn.split('>>')[0]
                    r_mol = Chem.MolFromSmiles(o_reactant)
                    for atom in r_mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                    o_reactant = Chem.MolToSmiles(r_mol)
                    o_reactant = Chem.MolToSmiles(Chem.MolFromSmiles(o_reactant))
                    e_smiles = get_e_smiles_with_check(rxn)
                    n_reactant = merge_smiles_only(e_smiles)
                    n_reactant = Chem.MolToSmiles(Chem.MolFromSmiles(n_reactant))

                    if o_reactant == n_reactant:
                        rxn_class_list.append(f'class_{class_}')
                        e_smiles_list.append(e_smiles)
                    else:
                        # print(o_reactant)
                        # print(n_reactant)
                        pass
                except:
                    pass

        elif split == "test":
            idx_to_drop = [822, 1282, 1490, 1558, 2810, 3487, 4958]
            rows_to_drop = []
            for j in range(augtime):
                rows_to_drop += [j*5007 + i for i in idx_to_drop]
            df = df.drop(rows_to_drop)
            df = df.reset_index(drop = True)
            rxn_class_list = [f"class_{n}" for n in df['class']]
            e_smiles_list = process_map(get_e_smiles, tqdm(df['reactants>reagents>production'], 
                                        desc = "transforming into ReactSeq ..."), max_workers = 20)

        elif split == "val":
            idx_to_drop = [2302, 2527, 2950, 4368, 4863, 4890]
            rows_to_drop = []
            for j in range(augtime):
                rows_to_drop += [j*5001 + i for i in idx_to_drop]            
            df = df.drop(rows_to_drop)
            df = df.reset_index(drop = True)
            rxn_class_list = [f"class_{n}" for n in df['class']]
            e_smiles_list = process_map(get_e_smiles, tqdm(df['reactants>reagents>production'], 
                                        desc = "transforming into ReactSeq ..."), max_workers = 20)

        return  e_smiles_list, rxn_class_list


def main():
    parser = argparse.ArgumentParser(description='Preprocess UPSTO_50k data')
    parser.add_argument('-data', type=str, required=True, choices=['50k', 'mit'], help='Data file to preprocess')
    parser.add_argument('-split', type=str, required=True, choices=['train', 'val', 'test'], help='Data split to preprocess')
    parser.add_argument('-augtime', type=int, required=True, help='Number of augmentations, we set 100 for train and 20 for test')
    parser.add_argument('-rxn_class', type=bool, default=False, choices=[False, True], help='Unkown reaction type (False) or Given reaction type (True)')
    args = parser.parse_args()

    # Load Augmented data
    augmentated_df = augmentate_data(raw_df, args.augtime)

    # Get ReactSeq, Clean few data
    e_smiles_list, rxn_class_list = preprocess_data(augmentated_df, args.data, args.split, args.augtime)

    # Tokenization ReactSeq with single characters
    src_list, tgt_list = [i.split(">>>")[0] for i in e_smiles_list], [i.split(">>>")[1] for i in e_smiles_list]

    # Unknown rxn_class 
    if not args.rxn_class:
        src = [" ".join(list(s)) for s in src_list]
        tgt = [" ".join(list(t)) for t in tgt_list]
        src_file_path = f"./datasets/{args.data}_ReactSeq_with_prompt/aug_{args.augtime}_{args.split}/src_{args.split}.txt"
        tgt_file_path = f"./datasets/{args.data}_ReactSeq_with_prompt/aug_{args.augtime}_{args.split}/tgt_{args.split}.txt"

    # Write processed data into .txt 
    os.makedirs(os.path.dirname(src_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)

    with open(src_file_path, "w") as f:
        for line in src: 
            f.write(line+'\n')

    with open(tgt_file_path, "w") as f:
        for line in tgt:
            f.write(line+'\n')


if __name__ == '__main__':
    main()
