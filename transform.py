import argparse
from rdkit import Chem
from e_smiles import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def main():
    parser = argparse.ArgumentParser(description='Transform ReactSeq into reactant SMILES')
    parser.add_argument('-src', type=str, required=True, default="datasets/50k_ReactSeq/aug20_test/src_aug20_test.txt", help='source (products) file path')
    parser.add_argument('-tgt', type=str, required=True, default="output/tgt_50k_ReactSeq_aug100_train_aug20_test_infer.txt", help='target (ReactSeq) file path')
    parser.add_argument('-output', type=str, required=True, default="output/pred_reactants_50k_ReactSeq_aug100_train_aug20_test_infer.txt", help='output (reactant SMILES) file path')
    parser.add_argument('-n_best', type=int, default=10, help='')
    args = parser.parse_args()

    # Preds     
    product_lis = []
    reactseq_lis = []
    rxn_reactseq_lis = []

    with open(args.src) as f:
        for line in f.readlines():
            line = line.replace('\n','').replace(" ","")
            product_lis.extend([line]*args.n_best)

    with open(args.tgt) as f:
        for line in f.readlines():
            line = line.replace('\n','').replace(" ","")
            reactseq_lis.append(line)

    try:
        assert len(product_lis) == len(reactseq_lis)
    except:
        print(len(product_lis))
        print(len(reactseq_lis))
        product_lis = product_lis[:len(reactseq_lis)]

    for p, e in zip(product_lis, reactseq_lis):
        rxn_reactseq_lis.append(p + ">>>" + e)
    
    print(len(rxn_reactseq_lis))

    # Transform ReactSeq into reactant SMILES
    pred_reactants = process_map(merge_smiles, tqdm(rxn_reactseq_lis), max_workers = 40)          

    # Save
    with open(args.output, "w") as f:
        for r in pred_reactants:
            f.write(r + "\n")

if __name__ == '__main__':
    main()