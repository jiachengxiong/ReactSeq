import argparse
import os
import pandas as pd
import random
from rdkit import Chem
from tqdm import tqdm
from tqdm.contrib import tzip
from e_smiles import get_e_smiles, merge_smiles, get_edit_from_e_smiles, merge_smiles_only, get_e_smiles_with_check
from tqdm.contrib.concurrent import process_map  

def remove_amap_not_in_product(rxn_smi):
    """
    Corrects the atom map numbers of atoms only in reactants. 
    This correction helps avoid the issue of duplicate atom mapping
    after the canonicalization step.
    """
    r, p = rxn_smi.split(">>")

    pmol = Chem.MolFromSmiles(p)
    pmol_amaps = set([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
    max_amap = max(pmol_amaps) #Atoms only in reactants are labelled starting with max_amap

    rmol  = Chem.MolFromSmiles(r)

    for atom in rmol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        if amap_num not in pmol_amaps:
            atom.SetAtomMapNum(max_amap+1)
            max_amap += 1

    r_updated = Chem.MolToSmiles(rmol)
    rxn_smi_updated = r_updated + ">>" + p
    return rxn_smi_updated

def random_smiles_with_map(smiles_with_map):
    mol = Chem.MolFromSmiles(smiles_with_map)
    atom_map_lis = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    random_id = random.randint(0,len(atom_map_lis)-1)
    Chem.Kekulize(mol)
    Chem.MolToSmiles(mol,rootedAtAtom = random_id)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))

    mol_ordered = Chem.RenumberAtoms(mol, order )
    for i in range(len(order)) :
        atom = mol_ordered.GetAtomWithIdx(i)
        atom.SetAtomMapNum(atom_map_lis[order[i]])
    smiles = Chem.MolToSmiles(mol_ordered,canonical = False,kekuleSmiles=True)
    return smiles

def canonical_smiles_with_map(smiles_with_map):  #unnecessary
    mol = Chem.MolFromSmiles(smiles_with_map)
    atom_map_lis = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    Chem.Kekulize(mol)
    order = list(Chem.CanonicalRankAtoms(mol, includeChirality=True))
    
    mol_ordered = Chem.RenumberAtoms(mol, order )
    for i in range(len(order)) :
        atom = mol_ordered.GetAtomWithIdx(i)
        atom.SetAtomMapNum(atom_map_lis[order[i]])
    smiles = Chem.MolToSmiles(mol_ordered,canonical = False,kekuleSmiles=True)
    return smiles

def augmentate_data(df, aug_time):
    new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}
    
    # Preprocessing loop
    for time in tqdm(range(aug_time), desc = "Augtime"):
        for idx in tqdm(range(len(df)), desc = "AugProcess"):
            element = df.loc[idx]
            uspto_id, class_id, rxn_smi = element['id'], element['class'], element['reactants>reagents>production']

            rxn_smi_new = remove_amap_not_in_product(rxn_smi)
            r,p  = rxn_smi_new.split('>>')

            p = random_smiles_with_map(p)
            mol = Chem.MolFromSmiles(p,sanitize = False)
            old_map_lis = [a.GetAtomMapNum() for a in mol.GetAtoms()]
            for i in range(mol.GetNumAtoms()):
                mol.GetAtomWithIdx(i).SetAtomMapNum(i + 1)
            p = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True)
            p_mol = Chem.MolFromSmiles(p,sanitize = False)

            new_map_lis = [a.GetAtomMapNum() for a in p_mol.GetAtoms()]
            if new_map_lis != [i+1 for i in range(len(new_map_lis))]:
                print(p)
                break

            dic_old_new_map = dict(zip(old_map_lis,new_map_lis))

            mol = Chem.MolFromSmiles(r)
            for i in range(mol.GetNumAtoms()):
                atom =  mol.GetAtomWithIdx(i)
                if atom.GetAtomMapNum() in dic_old_new_map.keys():
                    atom.SetAtomMapNum(dic_old_new_map[atom.GetAtomMapNum()])
                else:
                    pass

            r = Chem.MolToSmiles(mol)
            r = canonical_smiles_with_map(r) #unnecessary

            rxn_smi_new = r + '>>' + p
            new_dict['id'].append(uspto_id)
            new_dict['class'].append(class_id)
            new_dict['reactants>reagents>production'].append(rxn_smi_new)

    new_df = pd.DataFrame.from_dict(new_dict)
    return new_df


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
            e_smiles_list = process_map(get_e_smiles, tqdm(df['reactants>reagents>production'], desc = "transforming into ReactSeq ..."), max_workers = 20)

        elif split == "val":
            idx_to_drop = [2302, 2527, 2950, 4368, 4863, 4890]
            rows_to_drop = []
            for j in range(augtime):
                rows_to_drop += [j*5001 + i for i in idx_to_drop]            
            df = df.drop(rows_to_drop)
            df = df.reset_index(drop = True)
            rxn_class_list = [f"class_{n}" for n in df['class']]
            e_smiles_list = process_map(get_e_smiles, tqdm(df['reactants>reagents>production'], desc = "transforming into ReactSeq ..."), max_workers = 20)

        return  e_smiles_list, rxn_class_list


def main():
    parser = argparse.ArgumentParser(description='Preprocess UPSTO_50k data')
    parser.add_argument('-data', type=str, required=True, choices=['50k', 'mit'], help='Data file to preprocess')
    parser.add_argument('-split', type=str, required=True, choices=['train', 'val', 'test'], help='Data split to preprocess')
    parser.add_argument('-augtime', type=int, required=True, help='Number of augmentations, we set 100 for train and 20 for test')
    parser.add_argument('-rxn_class', type=bool, default=False, choices=[False, True], help='Unkown reaction type (False) or Given reaction type (True)')
    args = parser.parse_args()

    # Load data
    raw_df = pd.read_csv(f'./datasets/{args.data}_raw/raw_{args.split}.csv')

    # Data Augmentation (Kekulized and Mapped rxn_smiles)
    augmentated_df = augmentate_data(raw_df, args.augtime)

    # Get ReactSeq, Clean few data
    e_smiles_list, rxn_class_list = preprocess_data(augmentated_df, args.data, args.split, args.augtime)

    # Tokenization ReactSeq with single characters
    src_list, tgt_list = [i.split(">>>")[0] for i in e_smiles_list], [i.split(">>>")[1] for i in e_smiles_list]

    # Unknown/Given rxn_class 
    if not args.rxn_class:
        src = [" ".join(list(s)) for s in src_list]
        tgt = [" ".join(list(t)) for t in tgt_list]
        src_file_path = f"./datasets/{args.data}_ReactSeq/aug_{args.augtime}_{args.split}/src_{args.split}.txt"
        tgt_file_path = f"./datasets/{args.data}_ReactSeq/aug_{args.augtime}_{args.split}/tgt_{args.split}.txt"

    elif args.rxn_class:
        src = [c + " " + " ".join(list(s)) for c,s in zip(rxn_class_list, src_list)]
        tgt = [" ".join(list(t)) for t in tgt_list]
        src_file_path = f"./datasets/{args.data}_ReactSeq_with_rxn_class/aug_{args.augtime}_{args.split}/src_{args.split}.txt"
        tgt_file_path = f"./datasets/{args.data}_ReactSeq_with_rxn_class/aug_{args.augtime}_{args.split}/tgt_{args.split}.txt"

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
