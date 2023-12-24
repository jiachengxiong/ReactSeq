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
