# E_Smiles

## Environments

Need two virutal environments (```opennmt3``` and ```rdkit2019```)

### Environment 1：opennmt3 （for training and inferencing）

```bash
conda create -n opennmt3 python==3.8
conda activate opennmt3
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0 numpy transformers pandas tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U OpenNMT-py
```

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 2.0 <2.1

### Environment 2：rdkit2019 (for data processing related to rdkit and lingo)

```bash
conda create -n rdkit2019 python==3.7
conda activate rdkit2019
conda install -c rdkit rdkit=2019.03.2 -y
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple epam.indigo
pip install ipykernel --upgrade
```

rdkit2019 requires:

- python <=3.7


## Data and Preprocessing

The ```USPTO_50K``` raw data are stored in

```bash
data/50k_raw
```

## Generate E-SMILES

For mapped and kekulized rxn_smiles, we can get their corresponding E-SMILES.

Here is an example:

```bash
mapped_rxn: 
[CH:5]1=[C:1]([C:2]([CH3:3])=[O:4])[CH:9]=[C:8]2[C:7](=[CH:6]1)[NH:12][CH:11]=[CH:10]2.[O:20]([C:21]([O:22][C:23]([CH3:24])([CH3:26])[CH3:25])=[O:27])[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]>>[C:1]1([C:2]([CH3:3])=[O:4])=[CH:5][CH:6]=[C:7]2[C:8](=[CH:9]1)[CH:10]=[CH:11][N:12]2[C:13](=[O:14])[O:15][C:16]([CH3:17])([CH3:18])[CH3:19]
```

```bash
SMILES of Product: 
C1(C(C)=O)=CC=C2C(=C1)C=CN2C(=O)OC(C)(C)C
E-SMILES: 
C1(C(C)=O)=CC=C2C(=C1)C=CN2!C(=O)OC(C)(C)C<><C(OC(C)(C)C)(=O)[O:1]>
E-SMILES (rxn): 
C1(C(C)=O)=CC=C2C(=C1)C=CN2C(=O)OC(C)(C)C>>>C1(C(C)=O)=CC=C2C(=C1)C=CN2!C(=O)OC(C)(C)C<><C(OC(C)(C)C)(=O)[O:1]>
```

More details related to generating E-SMILES can be found in [Usage_Example_of_E_SMILES.ipynb](https://github.com/jiachengxiong/E_Smiles/blob/main/Usage_Example_of_E_SMILES.ipynb)

