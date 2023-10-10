# E_Smiles

## 环境

涉及两套环境(opennmt3 和 rdkit2019)

### 环境一：opennmt3 （用于训练和推理）

```bash
conda create -n opennmt3 python==3.8
conda activate opennmt3
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.0 numpy transformers pandas tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U OpenNMT-py
```

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 2.0 <2.1

### 环境二：rdkit2019 (用于数据处理、E-SMILES拼接回SMILES、TTA)

```bash
conda create -n rdkit2019 python==3.7
conda activate rdkit2019
conda install -c rdkit rdkit=2019.03.2 -y
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple epam.indigo
pip install ipykernel --upgrade
```

rdkit2019requires

- python <=3.7
