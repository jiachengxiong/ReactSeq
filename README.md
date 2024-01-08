# E_Smiles

## :anchor: Environments

Need two virtual environments (```opennmt3``` and ```rdkit2019```)

### Environment 1：opennmt3（for training and inferencing）

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


## 🚀 Quick Start of Generating E-SMILES

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

More details related to ```generating E-SMILES``` and ```transforming E-SMILES to SMILES of Reactants``` can be found in [Usage_Example_of_E_SMILES.ipynb](https://github.com/jiachengxiong/E_Smiles/blob/main/Usage_Example_of_E_SMILES.ipynb)


## 🛠️ Data and Preprocessing

The ```USPTO_50K``` raw data are sourced from [typed_schneider50k](https://github.com/Hanjun-Dai/GLN) and stored in

```bash
data/50k_raw
```

You can generate the augmentated data by using 
```
python preprocess_data.py -data 50k -split train -augtime 100 -rxn_class False
python preprocess_data.py -data 50k -split val -augtime 20 -rxn_class False
python preprocess_data.py -data 50k -split test -augtime 20 -rxn_class False
```
Note: It is suggested to process data under ```rdkit2019``` environment (rdkit version: 2019.03.2)

The processed data will be stored in 
```
data/50k_e_smiles/aug100_train
data/50k_e_smiles/aug20_val
data/50k_e_smiles/aug20_test
```

You can also download our pre-processed data from [google_drive](https://drive.google.com/drive/folders/1a6NL5apcP_7isY3HccLjkSsjJGwp_FwD?usp=sharing) and put them into the above place.

## Training
Before training, check out the settings in ```train.sh``` and corresponding ```.yml``` file in ```./config```. Then, run
```bash
bash train.sh
```

## Inferencing
Before inferencing, check out the settings in ```inference.sh``` and corresponding ```.yml``` file in ```./config```. Then, run
```bash
bash inference.sh
```
Here, we inference the test set (augtime x20) by our model.

## Transforming
The predictions of model are in the format of E-SMILES, need to be transformed to SMILES of reactants.

```bash
conda activate rdkit2019
python transform.py \
    -src "datasets/50k_e_smiles/aug20_test/src_aug20_test.txt" \
    -tgt "output/tgt_50k_e_smiles_aug100_train_aug20_test_infer.txt" \
    -output "output/pred_reactants_50k_e_smiles_aug100_train_aug20_test_infer.txt"
```
Note: Transform need to be under ```rdkit2019``` environment.

## Calculating Top-k Accuracy
Run  ```cal_top_k_accuracy.ipynb```.

## :fire: Quick Retrosynthesis Prediction
Please download our trained model from [google_drive](https://drive.google.com/drive/folders/1a6NL5apcP_7isY3HccLjkSsjJGwp_FwD?usp=sharing) and put them into `trained_models`.


## 🙌 Acknowledgments
Special thanks to [GraphRetro](https://github.com/vsomnath/graphretro) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for the code used in this project.
