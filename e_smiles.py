import numpy as np
from rdkit import Chem
import os
import sys
import copy
import re
from typing import List, Any
from indigo import * 
indigo = Indigo() 
import rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
print('rdkit version:' + rdkit.__version__)


BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],}



def get_bond_info(mol: Chem.Mol):
    """Get information on bonds in the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()
        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [bond.GetBondTypeAsDouble(), bond.GetIdx()]

    return bond_info


def map_reac_and_frag(reac_mols: List[Chem.Mol], frag_mols: List[Chem.Mol]):
    """Aligns reactant and fragment mols by computing atom map overlaps.

    Parameters
    ----------
    reac_mols: List[Chem.Mol],
        List of reactant mols
    frag_mols: List[Chem.Mol],
        List of fragment mols
    """
    if len(reac_mols) != len(frag_mols):
        return reac_mols, frag_mols
    reac_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in reac_mols]
    frag_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in frag_mols]

    overlaps = {i: [] for i in range(len(frag_mols))}
    for i, fmap in enumerate(frag_maps):
        overlaps[i].extend([len(set(fmap).intersection(set(rmap))) for rmap in reac_maps])
        overlaps[i] = overlaps[i].index(max(overlaps[i]))

    new_frag = [Chem.Mol(mol) for mol in frag_mols]
    new_reac = [Chem.Mol(reac_mols[overlaps[i]]) for i in overlaps]
    return new_reac, new_frag


def remove_s_H(frag_mol):
    while True:
        idx = ''
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
                idx= atom.GetIdx()

        if  idx != '' :     
            edit_mol = Chem.RWMol(frag_mol)
            edit_mol.RemoveAtom(idx)
            frag_mol = edit_mol.GetMol()
        else:
            break
    
    return frag_mol



def apply_edits_to_mol_change(mol, edits):
    """Apply edits to molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object
    edits: Iterable[str],
        Iterable of edits to apply. An edit is structured as a1:a2:b1:b2, where
        a1, a2 are atom maps of participating atoms and b1, b2 are previous and
        new bond orders. When  a2 = 0, we update the hydrogen count.
    """
    new_mol = Chem.RWMol(mol)
    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}

    for edit in edits:
        x, y, prev_bo, new_bo = edit.split(":")
        x, y = int(x), int(y)
        new_bo = float(new_bo)
        
        bond = new_mol.GetBondBetweenAtoms(amap[x],amap[y])

        if new_bo > 0:
            if bond is not None:
                new_mol.RemoveBond(amap[x],amap[y])
                new_mol.AddBond(amap[x],amap[y],BOND_FLOAT_TO_TYPE[new_bo])
                atom_x,atom_y = new_mol.GetAtomWithIdx(amap[x]),new_mol.GetAtomWithIdx(amap[y])
                
                try:
                    atom_x.SetNumExplicitHs(int(atom_x.GetNumExplicitHs()+ float(prev_bo)-float(new_bo)))
                except:
                    atom_x.SetNumExplicitHs(0)
                try:
                    atom_y.SetNumExplicitHs(int(atom_y.GetNumExplicitHs()+ float(prev_bo)-float(new_bo)))
                except:
                    atom_y.SetNumExplicitHs(0)
        
    pred_mol = new_mol.GetMol()
    return pred_mol


def apply_edits_to_mol_break(mol, edits):
    """Apply edits to molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object
    edits: Iterable[str],
        Iterable of edits to apply. An edit is structured as a1:a2:b1:b2, where
        a1, a2 are atom maps of participating atoms and b1, b2 are previous and
        new bond orders. When  a2 = 0, we update the hydrogen count.
    """
    mol = Chem.AddHs(mol)  
    Chem.Kekulize(mol)
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    new_mol = Chem.RWMol(mol)
    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}


    for edit in edits:
        x, y, prev_bo, new_bo = edit.split(":")
        x, y = int(x), int(y)
        new_bo = float(new_bo)

        if y == 0:  
            cent_atom = mol.GetAtomWithIdx(amap[x])
            for neibor in cent_atom.GetNeighbors():
                if neibor.GetAtomicNum() == 1:
                    new_mol.RemoveBond(amap[x],neibor.GetIdx())
                    break
                else:
                    pass

        elif y != 0:
            bond = new_mol.GetBondBetweenAtoms(amap[x],amap[y])
            if bond is not None:
                new_mol.RemoveBond(amap[x],amap[y])
        
    pred_mol = new_mol.GetMol()
    pred_mol = Chem.RemoveHs(pred_mol,sanitize = False)
    
    return pred_mol



def find_reac_edit(frag_mols_1,reac_mols_1,core_edits):
    reac_mol_map_num = [i.GetAtomMapNum() for i in reac_mols_1.GetAtoms()] 
    frag_mol_map_num = [i.GetAtomMapNum() for i in frag_mols_1.GetAtoms()]
    lg_map_num = [i for i in reac_mol_map_num if i not in frag_mol_map_num]  
    attach_map_num = 0  
    
    reac_edit = []
    

    core_edits = core_edits + [':'.join([i.split(':')[1],i.split(':')[0],i.split(':')[2],i.split(':')[3]]) for i in core_edits]
    

    for core_edit in core_edits:   
        core_edit_ = core_edit.split(':')  
        if float(core_edit_[3]) == 0 and int(core_edit_[0]) in frag_mol_map_num:  
            attach_map_num = int(core_edit_[0])
        elif float(core_edit_[2]) - float(core_edit_[3]) > 0 and int(core_edit_[0]) in frag_mol_map_num:
            attach_map_num = int(core_edit_[0])
            
            
        else:
            continue

        if str(attach_map_num) != '0' and str(attach_map_num) != core_edit_[0]:   
            continue
        
        
        frag_mols_1_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in frag_mols_1.GetAtoms()}
        reac_mols_1_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mols_1.GetAtoms()}

        frag_attach_H = frag_mols_1.GetAtomWithIdx(frag_mols_1_amap[attach_map_num]).GetNumExplicitHs()
        reac_attach_H = reac_mols_1.GetAtomWithIdx(reac_mols_1_amap[attach_map_num]).GetNumExplicitHs()

        frag_attach_charge = frag_mols_1.GetAtomWithIdx(frag_mols_1_amap[attach_map_num]).GetFormalCharge()
        reac_attach_charge = reac_mols_1.GetAtomWithIdx(reac_mols_1_amap[attach_map_num]).GetFormalCharge()
        
        
        if lg_map_num != []:
            for bond in reac_mols_1.GetBonds():
                EndMapNum = bond.GetEndAtom().GetAtomMapNum()
                BeginMapNum = bond.GetBeginAtom().GetAtomMapNum()

                if (BeginMapNum == attach_map_num) and (EndMapNum in lg_map_num):   
                    reac_edit.append("{}:{}:{}:{}".format(BeginMapNum,EndMapNum,bond.GetBondTypeAsDouble(),0.0))
                elif (EndMapNum == attach_map_num) and (BeginMapNum in lg_map_num):

                    reac_edit.append("{}:{}:{}:{}".format(EndMapNum,BeginMapNum,bond.GetBondTypeAsDouble(),0.0))


        
        
        elif lg_map_num == []:

            
            if Chem.MolToSmiles(reac_mols_1) == Chem.MolToSmiles(frag_mols_1):
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,0.0)) 
            if (reac_attach_H - frag_attach_H) == 1 and (reac_attach_charge - frag_attach_charge) == 0:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,1.0,0.0))
            if (reac_attach_H - frag_attach_H) == 2 and (reac_attach_charge - frag_attach_charge) == 0:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,2.0,0.0)) 
                
        if (reac_attach_charge - frag_attach_charge)  == -1:
            if "{}:{}:{}:{}".format(attach_map_num,0,0.0,-1.0) not in reac_edit:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,-1.0))  

        if (reac_attach_charge - frag_attach_charge) == 1:
            if "{}:{}:{}:{}".format(attach_map_num,0,0.0,1.0) not in reac_edit:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,1.0))  
                

    return reac_edit





def correct_mol_1(mol,is_nitrine_c):
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():

        if is_nitrine_c == True and atom.GetAtomicNum() == 7 and sum([i.GetBondTypeAsDouble() for i in atom.GetBonds()]) == 4 and 1.5 not in [i.GetBondTypeAsDouble() for i in atom.GetBonds()] and atom.GetFormalCharge()==0: #调整N的电荷
            atom.SetFormalCharge(1)
        else:
            pass
        
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)

    return mol


def correct_mol(mol_,keep_map):

    mol = copy.deepcopy(mol_)
    atom_map_lis = [] 
    idx_H_dic = {}
    
    for atom in mol.GetAtoms():
        atom_map_lis.append(atom.GetAtomMapNum())

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and sum([i.GetBondTypeAsDouble() for i in atom.GetBonds()]) == 4 and 1.5 not in [i.GetBondTypeAsDouble() for i in atom.GetBonds()] and atom.GetFormalCharge()==0: #调整N的电荷
            pass
        elif atom.GetAtomicNum() == 15 and atom.GetExplicitValence() == 5 and 1.5 not in [i.GetBondTypeAsDouble() for i in atom.GetBonds()] and atom.GetFormalCharge()==0: #调整N的电荷
            idx_H_dic[atom.GetIdx()] = atom.GetNumExplicitHs()   
        else:
            pass
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(False)
        atom.SetAtomMapNum(0)

    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
        
        
    temp = Chem.MolToMolBlock(mol,kekulize = True)  
    mol = Chem.MolFromMolBlock(temp,removeHs = False,sanitize= False)
    

    
    if keep_map:
        for i in range(0,mol.GetNumAtoms()):
 
            mol.GetAtomWithIdx(i).SetAtomMapNum(atom_map_lis[i])
            if i in idx_H_dic.keys():
                
                mol.GetAtomWithIdx(i).SetNoImplicit(True)
                mol.GetAtomWithIdx(i).SetNumExplicitHs(idx_H_dic[i])
                
    
    for i in range(0,mol.GetNumAtoms()):
        mol.GetAtomWithIdx(i).SetChiralTag(mol_.GetAtomWithIdx(i).GetChiralTag())
        
                
    n_Chirals = Chem.FindMolChiralCenters(mol) 

    return mol


def get_atom_map_chai_dic(mol):
    dic = {}
    for idx,chiral in Chem.FindMolChiralCenters(mol):
        atom_map = mol.GetAtomWithIdx(idx).GetAtomMapNum()
        dic[atom_map] = chiral
    return dic


def get_atom_map_stereo_dic(mol):
    map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    stereo_dic = {}
    for bond in mol.GetBonds():
        b_map,e_map =  map_a[bond.GetBeginAtomIdx()],map_a[bond.GetEndAtomIdx()]
        stereo_dic[tuple(sorted([b_map,e_map]))] = bond.GetStereo()
    return stereo_dic


def cano_smiles_map(smiles):
    atom_map_lis = []
    mol = Chem.MolFromSmiles(smiles,sanitize = False)
    for atom in mol.GetAtoms():
        atom_map_lis.append(atom.GetAtomMapNum())
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True)
    mol = Chem.MolFromSmiles(smiles,sanitize = False)    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom_map_lis[atom.GetIdx()])
    smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True) 
    return smiles



def get_stereo_edit_mine(reac_mol,prod_mol):   

    reac_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()}
    prod_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    
    
    for atom in reac_mol.GetAtoms():
        atom.SetAtomMapNum(0)
   
    
    r_rank = list(Chem.CanonicalRankAtoms(reac_mol, breakTies=False))
    r_idx = [i for i in range(reac_mol.GetNumAtoms())]
    dic_idx_rank  = dict(zip(r_idx,r_rank))


    p_stereo_dic = {}
    for bond in prod_mol.GetBonds():
        b_map,e_map =  prod_map_a[bond.GetBeginAtomIdx()],prod_map_a[bond.GetEndAtomIdx()]
        p_stereo_dic[tuple(sorted([b_map,e_map]))] = bond.GetStereo()

    r_stereo_dic = {}
    for bond in reac_mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            
            b_atom,e_atom = bond.GetBeginAtom(),bond.GetEndAtom()

            b_neis = b_atom.GetNeighbors()
            b_neis = [i for i in b_neis if i.GetIdx() != e_atom.GetIdx()]
            b_neis_rank = [dic_idx_rank[i.GetIdx()] for i in b_neis]

            e_neis = e_atom.GetNeighbors()
            e_neis = [i for i in e_neis if i.GetIdx() != b_atom.GetIdx()]
            e_neis_rank = [dic_idx_rank[i.GetIdx()] for i in e_neis]

            
            b_neis_rank = b_neis_rank + ['H'] *  (2 - len(b_neis_rank))
            e_neis_rank = e_neis_rank + ['H'] *  (2 - len(e_neis_rank))
            
            if len(b_neis_rank) == len(set(b_neis_rank)) and len(e_neis_rank) == len(set(e_neis_rank)):
            
                b_map,e_map =  reac_map_a[bond.GetBeginAtomIdx()],reac_map_a[bond.GetEndAtomIdx()]
                r_stereo_dic[tuple(sorted([b_map,e_map]))] = bond.GetStereo()
        else:
            pass

    stereo_edits = []
    for atom_pair,stereo in r_stereo_dic.items():
        if atom_pair in p_stereo_dic.keys() and stereo != p_stereo_dic[atom_pair]:
            if stereo == Chem.rdchem.BondStereo.STEREONONE:
                stereo = 'a'
            elif stereo == Chem.rdchem.BondStereo.STEREOE:
                stereo = 'e'   
            elif stereo == Chem.rdchem.BondStereo.STEREOZ:
                stereo = 'z'   
            stereo_edits.append('{}:{}:{}:{}'.format(atom_pair[0],atom_pair[1],0,stereo))
    return stereo_edits



def apply_stereo_change(prod_mol,stereo_edits):
    p_amap_idx =  {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}

    prod_mol = copy.deepcopy(prod_mol)
    

    prod_mol_t = copy.deepcopy(prod_mol)

    for stereo_edit in stereo_edits:

        b_map = int(stereo_edit.split(':')[0])
        e_map = int(stereo_edit.split(':')[1])
 
        b_n = prod_mol.GetAtomWithIdx(p_amap_idx[b_map]).GetNeighbors()
        b_n = [i.GetAtomMapNum() for i in b_n]
        b_n = [i for i in b_n if i not in [b_map,e_map]]
        
        e_n = prod_mol.GetAtomWithIdx(p_amap_idx[e_map]).GetNeighbors()
        e_n = [i.GetAtomMapNum() for i in e_n]
        e_n = [i for i in e_n if i not in [b_map,e_map]]
        

        
        f_b_n = b_n[0]
        m_cip_rank = 0
        for i in b_n[:]:

            c_cip_rank = int(prod_mol_t.GetAtomWithIdx(p_amap_idx[i]).GetProp('_CIPRank'))

            if c_cip_rank >= m_cip_rank:
                f_b_n = i
                m_cip_rank = c_cip_rank

                
        f_e_n = e_n[0]
        m_cip_rank = 0
        for i in e_n[:]:

            c_cip_rank = int(prod_mol_t.GetAtomWithIdx(p_amap_idx[i]).GetProp('_CIPRank'))

            if c_cip_rank >= m_cip_rank:
                f_e_n = i
                m_cip_rank = c_cip_rank

        
        if stereo_edit[-2:] == ':e':

            bond = prod_mol.GetBondBetweenAtoms(p_amap_idx[b_map],p_amap_idx[e_map])
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
            

            try:
                bond.SetStereoAtoms(p_amap_idx[f_b_n],p_amap_idx[f_e_n])
            except:
                bond.SetStereoAtoms(p_amap_idx[f_e_n],p_amap_idx[f_b_n])



                
        if stereo_edit[-2:] == ':z':
            bond = prod_mol.GetBondBetweenAtoms(p_amap_idx[b_map],p_amap_idx[e_map])
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
            try:
                bond.SetStereoAtoms(p_amap_idx[f_b_n],p_amap_idx[f_e_n])
            except:
                bond.SetStereoAtoms(p_amap_idx[f_e_n],p_amap_idx[f_b_n])
                
                
        elif stereo_edit[-2:] == ':a':
            bond = prod_mol.GetBondBetweenAtoms(p_amap_idx[b_map],p_amap_idx[e_map])
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY)
            
    return  prod_mol


def add_Cl(mol):
    add_Cl_atom_idx = []
    for atom in mol.GetAtoms():
        Double_O_count = 0
        if atom.GetAtomicNum() == 16 and sorted([i.GetBondTypeAsDouble() for i in atom.GetBonds()]) == [1,2,2]:
            neibors = atom.GetNeighbors()
            for neibor in neibors:
                if neibor.GetAtomicNum() == 8:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(),neibor.GetIdx())
                    if bond.GetBondTypeAsDouble() == 2:
                        Double_O_count += 1
                    else:
                        pass
                else:
                    pass
            if Double_O_count == 2:
                add_Cl_atom_idx.append(atom.GetIdx())
                
    if len(add_Cl_atom_idx) == 1:
        map_lis = [i.GetAtomMapNum() for i in mol.GetAtoms()]
        mw = Chem.RWMol(mol)
        mw.AddAtom(Chem.Atom(17))

        mw.GetAtomWithIdx(len(map_lis)).SetAtomMapNum(max(map_lis)+1)
        mw.AddBond(add_Cl_atom_idx[0],len(map_lis), BOND_FLOAT_TO_TYPE[1])
        mol =  mw.GetMol() 
        
    return mol



def neu_sulf_charge(mol):

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == -1:

            neibors = atom.GetNeighbors()
            if len(neibors) == 1 and neibors[0].GetAtomicNum() == 16 and neibors[0].GetExplicitValence() == 4:
                atom.SetFormalCharge(0) 
            else:
                pass
            
    return mol



def align_kekule_pairs(r: str, p: str) :
    """Aligns kekule pairs to ensure unchanged bonds have same bond order in
    previously aromatic rings.

    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    """
    reac_mol = Chem.MolFromSmiles(r)
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap = max_amap + 1

    prod_mol = Chem.MolFromSmiles(p)

    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)

    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)

    
    reac_edit = {}
    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            if reac_new[bond][0] != prod_new[bond][0] or reac_prev[bond][0] == 1.5:
                reac_new[bond][0] = prod_new[bond][0]
                reac_edit[bond] = reac_new[bond]

            

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in reac_edit:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol


def count_kekule_d(r,p):
    prod_mol = Chem.MolFromSmiles(p)
    prod_s = get_bond_info(prod_mol)

    prod_mol = Chem.MolFromSmiles(p,sanitize = False)
    prod_k = get_bond_info(prod_mol)
    
    reac_mol = Chem.MolFromSmiles(r)
    reac_s = get_bond_info(reac_mol)

    reac_mol = Chem.MolFromSmiles(r,sanitize = False)
    reac_k = get_bond_info(reac_mol)
    
    d_count = 0
    for pair in reac_s.keys():
        if pair in prod_s.keys():
            if reac_s[pair][0] == prod_s[pair][0]:
                if reac_k[pair][0] != prod_k[pair][0]:
                    d_count += 1
    
    return d_count


def get_kekule_aligned_r(r,p):
    if count_kekule_d(r,p) == 0:
        return r
    else:
        
        min_r_s_lis = []
        for r_s in r.split('.'):
        
            min_count = 1000
            min_r_s = ''

            mol = Chem.MolFromSmiles(r_s)
            suppl = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL)

            for i in range(len(suppl)):
                r_s = Chem.MolToSmiles(suppl[i],kekuleSmiles = True)
                count = count_kekule_d(r_s,p)
                if count <= min_count:
                    min_r_s = r_s
                    min_count = count
                    
            min_r_s_lis.append(min_r_s)
                
    return '.'.join(min_r_s_lis)


def apply_edits_to_mol_connect(mol, edits):
    """Apply edits to molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object
    edits: Iterable[str],
        Iterable of edits to apply. An edit is structured as a1:a2:b1:b2, where
        a1, a2 are atom maps of participating atoms and b1, b2 are previous and
        new bond orders. When  a2 = 0, we update the hydrogen count.
    """
    new_mol = Chem.RWMol(mol)
    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}


    for edit in edits:
        x, y, prev_bo, new_bo = edit.split(":")
        x, y = int(x), int(y)
        new_bo = float(new_bo)


        new_mol.AddBond(amap[x],amap[y],BOND_FLOAT_TO_TYPE[new_bo])
        
    pred_mol = new_mol.GetMol()
    
    return pred_mol


def get_charge_edit_mine(reac_mol, prod_mol,core_edits):
    
    lg_site_lis = []
    for core_edit in core_edits:
        x,y,bo,n_bo = core_edit.split(':')
        if float(bo) - float(n_bo) > 0:
            lg_site_lis.append(int(x))
            lg_site_lis.append(int(y))
    lg_site_lis = [i for i in lg_site_lis if i != 0]
    
    dict_reac_charges = {}
    for atom in reac_mol.GetAtoms():
        dict_reac_charges[atom.GetAtomMapNum()] = atom.GetFormalCharge()

    dict_prod_charges = {}
    for atom in prod_mol.GetAtoms():
        dict_prod_charges[atom.GetAtomMapNum()] = atom.GetFormalCharge()
    
    charge_edits = []
    for atom_map, charge in dict_prod_charges.items():
        if atom_map in dict_reac_charges.keys():
            if dict_reac_charges[atom_map] != charge and atom_map not in lg_site_lis:  
                edit = f"{atom_map}:{0}:{0}:{dict_reac_charges[atom_map]}"  
                charge_edits.append(edit)
                
    return charge_edits




def get_atom_map_charge_dic(mol):
    dic = {}
    for atom in mol.GetAtoms():
        dic[atom.GetAtomMapNum()] = atom.GetFormalCharge()
    return dic


def apply_charge_change(mol,charge_edits):

    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    for edit in charge_edits:
        x, y, prev_charge, new_charge = edit.split(":")
        mol.GetAtomWithIdx(amap[int(x)]).SetFormalCharge(int(new_charge))
    return mol


def get_core_edit_mine(reac_mol, prod_mol):

    prod_bonds = get_bond_info(prod_mol)
    reac_bonds = get_bond_info(reac_mol)
    
    rxn_core_break = set()
    rxn_core_lack = set()
    rxn_core = set()
    core_edits = []

    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}
    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in prod_bonds:
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            prod_bo, reac_bo = prod_bonds[bond][0], reac_bonds[bond][0]

            a_start, a_end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            prod_bo = prod_bonds[bond][0]

            start, end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])
            rxn_core_break.update([a_start, a_end])

    for bond in reac_bonds:
        if bond not in prod_bonds:
            amap1, amap2 = bond
            rxn_core_lack.update([amap1, amap2])
            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):   
                a_start, a_end = sorted([amap1, amap2])
                reac_bo = reac_bonds[bond][0]
                edit = f"{a_start}:{a_end}:{0.0}:{reac_bo}"
                core_edits.append(edit)
                rxn_core.update([a_start, a_end])
                

    if True:
        reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}
        for atom in prod_mol.GetAtoms():
            amap_num = atom.GetAtomMapNum()
            if (amap_num in rxn_core_break) or (amap_num not in rxn_core_lack):
                pass
            else:
                amap_num = atom.GetAtomMapNum()
                numHs_prod = atom.GetTotalNumHs()
                numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()
                if numHs_prod != numHs_reac:
                    edit = f"{amap_num}:{0}:{1.0}:{0.0}"
                    core_edits.append(edit)
                    rxn_core.add(amap_num)
                    
        
    return core_edits



def get_chai_edit_mine(reac_mol, prod_mol):    
    reac_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()}
    prod_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    
    reac_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(reac_mol,includeUnassigned=True):
        reac_ChiralCenters.append((reac_map_a[ChiralCenters[0]],ChiralCenters[1]))

    prod_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(prod_mol,includeUnassigned=True):
        prod_ChiralCenters.append((prod_map_a[ChiralCenters[0]],ChiralCenters[1]))
        
    dict_reac_ChiralCenters = dict(reac_ChiralCenters)
    dict_prod_ChiralCenters = dict(prod_ChiralCenters)
    
    
    chai_edits = []

    for amap_num,chiral in dict_prod_ChiralCenters.items():
        if amap_num in dict_reac_ChiralCenters.keys():
            if chiral != dict_reac_ChiralCenters[amap_num]:
                edit = f"{amap_num}:{0}:{0}:{dict_reac_ChiralCenters[amap_num]}"
                chai_edits.append(edit)
        else:
            pass
    
    for amap_num,chiral in dict_reac_ChiralCenters.items():  
        if (amap_num not in dict_prod_ChiralCenters.keys()) and (amap_num in prod_map_a.values()) and chiral != '?':
            edit = f"{amap_num}:{0}:{0}:{chiral}"
            chai_edits.append(edit)
        
    return chai_edits





def get_chai_edit_mine(reac_mol, prod_mol):    
    reac_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()}
    prod_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    
    reac_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(reac_mol,includeUnassigned=True):
        reac_ChiralCenters.append((reac_map_a[ChiralCenters[0]],ChiralCenters[1]))

    prod_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(prod_mol,includeUnassigned=True):
        prod_ChiralCenters.append((prod_map_a[ChiralCenters[0]],ChiralCenters[1]))
        
    dict_reac_ChiralCenters = dict(reac_ChiralCenters)
    dict_prod_ChiralCenters = dict(prod_ChiralCenters)
    
    
    chai_edits = []

    for amap_num,chiral in dict_prod_ChiralCenters.items():
        if amap_num in dict_reac_ChiralCenters.keys():
            if chiral != dict_reac_ChiralCenters[amap_num]:
                edit = f"{amap_num}:{0}:{0}:{dict_reac_ChiralCenters[amap_num]}"
                chai_edits.append(edit)
        else:
            pass
    
    for amap_num,chiral in dict_reac_ChiralCenters.items(): 
        if (amap_num not in dict_prod_ChiralCenters.keys())and (amap_num in prod_map_a.values()):
            edit = f"{amap_num}:{0}:{0}:{chiral}"

            chai_edits.append(edit)
        
    return chai_edits



def get_chai_edit_mine(reac_mol, prod_mol):    
    reac_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()}
    prod_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    
    reac_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(reac_mol,includeUnassigned=True):
        reac_ChiralCenters.append((reac_map_a[ChiralCenters[0]],ChiralCenters[1]))

    prod_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(prod_mol,includeUnassigned=True):
        prod_ChiralCenters.append((prod_map_a[ChiralCenters[0]],ChiralCenters[1]))
        
    dict_reac_ChiralCenters = dict(reac_ChiralCenters)
    dict_prod_ChiralCenters = dict(prod_ChiralCenters)
    
    
    chai_edits = []

    for amap_num,chiral in dict_prod_ChiralCenters.items():
        if amap_num in dict_reac_ChiralCenters.keys():
            if chiral != dict_reac_ChiralCenters[amap_num]:
                edit = f"{amap_num}:{0}:{0}:{dict_reac_ChiralCenters[amap_num]}"
                chai_edits.append(edit)
        else:
            pass
    
    for amap_num,chiral in dict_reac_ChiralCenters.items(): 
        if (amap_num not in dict_prod_ChiralCenters.keys()) and (amap_num in prod_map_a.values()) and chiral != '?':
            edit = f"{amap_num}:{0}:{0}:{chiral}"
            chai_edits.append(edit)
        
    return chai_edits




def get_lg_map_lis(frag_mols,reac_mols,core_edits,prod_mol):   
    
    lg_map_lis = []
    prod_map_num_lis = [i.GetAtomMapNum() for i in prod_mol.GetAtoms()]
    
    for frag_mols_1,reac_mols_1 in zip(frag_mols[:],reac_mols[:]):
        reac_edits = find_reac_edit(frag_mols_1,reac_mols_1,core_edits)  
        

        reac_edits_a = []
        reac_edits_b = []
        for reac_edit in reac_edits:
            if reac_edit[:3] == '0:0': 
                reac_edits_a.append(reac_edit)
            elif reac_edit[-7:] == '0.0:0.0':    
                reac_edits_a.append(reac_edit)
            elif reac_edit[-10:] == '0:0.0:-1.0':   
                reac_edits_a.append(reac_edit)
            elif reac_edit[-9:] == '0:0.0:1.0':    
                reac_edits_a.append(reac_edit)
            else:
                reac_edits_b.append(reac_edit)
        

        for reac_edit in reac_edits_a:
            if reac_edit[:3] == '0:0':  
                pass
            elif reac_edit[-7:] == '0.0:0.0':
                pass
            elif reac_edit[-10:] == '0:0.0:-1.0':
                edit_map_num_lis = reac_edit.split(':')[:2]
                attach_map_num_1 = [int(i) for i in edit_map_num_lis if int(i) in prod_map_num_lis] 
                lg_smiles = '-1.0'
                lg_map_lis.append((lg_smiles,attach_map_num_1))
            elif reac_edit[-9:] == '0:0.0:1.0':
                edit_map_num_lis = reac_edit.split(':')[:2]
                attach_map_num_1 = [int(i) for i in edit_map_num_lis if int(i) in prod_map_num_lis] 
                lg_smiles = '1.0'
                lg_map_lis.append((lg_smiles,attach_map_num_1))
        

        frag_1_map_num_lis = [i.GetAtomMapNum() for i in frag_mols_1.GetAtoms() if i.GetAtomMapNum() != 0]
        reac_frag_mol = apply_edits_to_mol_break(reac_mols_1 , reac_edits_b)
        reac_frag_mols = Chem.GetMolFrags(reac_frag_mol,asMols=True,sanitizeFrags = False)
        
        
        reac_edit_added = []
        for reac_frag_mol in reac_frag_mols[:]:

            reac_frag_map_num_lis = [i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms() if i.GetAtomMapNum() != 0]

            if set(reac_frag_map_num_lis) == set(frag_1_map_num_lis): 
                pass
            else:
                attach_map_num_1 = []          
                for reac_edit in reac_edits:    
                    if reac_edit in reac_edit_added:
                        continue
                    else:
                        pass

                    
                    b,e = int(reac_edit.split(':')[0]),int(reac_edit.split(':')[1]) 
                    if e in reac_frag_map_num_lis and b in frag_1_map_num_lis:

                        for atom in reac_frag_mol.GetAtoms():  
                            if atom.GetAtomMapNum() == int(e):
                                atom.SetAtomMapNum(500+atom.GetAtomMapNum())
                                break 
                            else:
                                pass
                        reac_edit_added.append(reac_edit)
                        
                        
                      
                        if len(attach_map_num_1) == 1:

                            if [str(attach_map_num_1[0]),str(atom.GetAtomMapNum()-500)] in [i.split(':')[:2] for i in reac_edits ]: #上一个合成子上的连接点和本离去基团的连接点配对
                                if atom.GetAtomMapNum() == max([i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms()]):
                                    attach_map_num_1 = [b] + attach_map_num_1
                                else:
                                    attach_map_num_1.append(b)
                            else: 
                                if atom.GetAtomMapNum() == max([i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms()]):
                                    attach_map_num_1.append(b)
                                else:
                                    attach_map_num_1 = [b] + attach_map_num_1
                        elif len(attach_map_num_1) == 0:
                            attach_map_num_1.append(b)
                            

                    else:
                        pass
                        
                    if reac_frag_mol.GetAtomWithIdx(0).GetAtomicNum() == 1 and len(attach_map_num_1) == 1:  
                        break

                
                lg_smiles = Chem.MolToSmiles(reac_frag_mol,kekuleSmiles = True) 
                lg = Chem.MolFromSmiles(lg_smiles)
                Chem.Kekulize(lg)
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() >= 500:
                        atom.SetAtomMapNum(1)
                        pass
                    else:
                        atom.SetAtomMapNum(0)
                lg_smiles = Chem.MolToSmiles(lg,canonical = False,kekuleSmiles = True)           
                
                if attach_map_num_1 != []:
                    lg_map_lis.append((lg_smiles,attach_map_num_1))
                    
    return lg_map_lis






def get_core_edit_mine(reac_mol, prod_mol):

    prod_bonds = get_bond_info(prod_mol)
    reac_bonds = get_bond_info(reac_mol)
    
    rxn_core_break = set()
    rxn_core_lack = set()
    rxn_core = set()
    core_edits = []

    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}
    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in prod_bonds:
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            prod_bo, reac_bo = prod_bonds[bond][0], reac_bonds[bond][0]

            a_start, a_end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            prod_bo = prod_bonds[bond][0]

            start, end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])
            rxn_core_break.update([a_start, a_end])

    for bond in reac_bonds:
        if bond not in prod_bonds:
            amap1, amap2 = bond
            rxn_core_lack.update([amap1, amap2])
            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):   
                a_start, a_end = sorted([amap1, amap2])
                reac_bo = reac_bonds[bond][0]
                edit = f"{a_start}:{a_end}:{0.0}:{reac_bo}"
                core_edits.append(edit)
                rxn_core.update([a_start, a_end])
                


    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        if (amap_num in rxn_core_break) or (amap_num not in rxn_core_lack):
            pass
        else:
            amap_num = atom.GetAtomMapNum()
            numHs_prod = atom.GetTotalNumHs()
            numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()
            if numHs_prod != numHs_reac:
                edit = f"{amap_num}:{0}:{1.0}:{0.0}"
                core_edits.append(edit)
                rxn_core.add(amap_num)
                
                
    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        if amap_num in rxn_core:
            pass
        else:
            amap_num = atom.GetAtomMapNum()
            Degree_prod = atom.GetDegree()
            Degree_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetDegree()

            if Degree_prod - Degree_reac == -1:
                edit = f"{amap_num}:{0}:{1.0}:{0.0}"
                core_edits.append(edit)
                rxn_core.add(amap_num)
                    
    
    
    
    
    return core_edits



def find_reac_edit(frag_mols_1,reac_mols_1,core_edits):
    reac_mol_map_num = [i.GetAtomMapNum() for i in reac_mols_1.GetAtoms()] 
    frag_mol_map_num = [i.GetAtomMapNum() for i in frag_mols_1.GetAtoms()]
    lg_map_num = [i for i in reac_mol_map_num if i not in frag_mol_map_num]  
    attach_map_num = 0  
    
    reac_edit = []
    
    core_edits = core_edits + [':'.join([i.split(':')[1],i.split(':')[0],i.split(':')[2],i.split(':')[3]]) for i in core_edits]
    

    for core_edit in core_edits:   
        core_edit_ = core_edit.split(':')  

        if float(core_edit_[3]) == 0 and int(core_edit_[0]) in frag_mol_map_num:  
            attach_map_num = int(core_edit_[0])
        elif float(core_edit_[2]) - float(core_edit_[3]) > 0 and int(core_edit_[0]) in frag_mol_map_num:
            attach_map_num = int(core_edit_[0])
            

        else:
            continue

        if str(attach_map_num) != '0' and str(attach_map_num) != core_edit_[0]: 
            continue
        
        
        frag_mols_1_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in frag_mols_1.GetAtoms()}
        reac_mols_1_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mols_1.GetAtoms()}

        frag_attach_H = frag_mols_1.GetAtomWithIdx(frag_mols_1_amap[attach_map_num]).GetNumExplicitHs()
        reac_attach_H = reac_mols_1.GetAtomWithIdx(reac_mols_1_amap[attach_map_num]).GetNumExplicitHs()

        frag_attach_charge = frag_mols_1.GetAtomWithIdx(frag_mols_1_amap[attach_map_num]).GetFormalCharge()
        reac_attach_charge = reac_mols_1.GetAtomWithIdx(reac_mols_1_amap[attach_map_num]).GetFormalCharge()
        
        
        if lg_map_num != []:
            for bond in reac_mols_1.GetBonds():
                EndMapNum = bond.GetEndAtom().GetAtomMapNum()
                BeginMapNum = bond.GetBeginAtom().GetAtomMapNum()
                if (BeginMapNum == attach_map_num) and (EndMapNum in lg_map_num):   
                    reac_edit.append("{}:{}:{}:{}".format(BeginMapNum,EndMapNum,bond.GetBondTypeAsDouble(),0.0))
                elif (EndMapNum == attach_map_num) and (BeginMapNum in lg_map_num):
                    reac_edit.append("{}:{}:{}:{}".format(EndMapNum,BeginMapNum,bond.GetBondTypeAsDouble(),0.0))


    
        elif lg_map_num == []:

            
            if Chem.MolToSmiles(reac_mols_1) == Chem.MolToSmiles(frag_mols_1):
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,0.0))  
            if (reac_attach_H - frag_attach_H) == 1 and (reac_attach_charge - frag_attach_charge) == 0:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,1.0,0.0))
            if (reac_attach_H - frag_attach_H) == 2 and (reac_attach_charge - frag_attach_charge) == 0:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,2.0,0.0)) 
                
        if (reac_attach_charge - frag_attach_charge)  == -1:
            if "{}:{}:{}:{}".format(attach_map_num,0,0.0,-1.0) not in reac_edit:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,-1.0))  
                
        if (reac_attach_charge - frag_attach_charge) == 1:
            if "{}:{}:{}:{}".format(attach_map_num,0,0.0,1.0) not in reac_edit:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,1.0))  
                

        if (reac_attach_charge - frag_attach_charge) == 2:
            if "{}:{}:{}:{}".format(attach_map_num,0,0.0,2.0) not in reac_edit:
                reac_edit.append("{}:{}:{}:{}".format(attach_map_num,0,0.0,2.0))  

                

    return reac_edit




def get_lg_map_lis(frag_mols,reac_mols,core_edits,prod_mol):   
    
    lg_map_lis = []
    prod_map_num_lis = [i.GetAtomMapNum() for i in prod_mol.GetAtoms()]
    
    for frag_mols_1,reac_mols_1 in zip(frag_mols[:],reac_mols[:]):
        reac_edits = find_reac_edit(frag_mols_1,reac_mols_1,core_edits)  
        

        reac_edits_a = []
        reac_edits_b = []
        for reac_edit in reac_edits:
            if reac_edit[:3] == '0:0':  
                reac_edits_a.append(reac_edit)
            elif reac_edit[-7:] == '0.0:0.0':   
                reac_edits_a.append(reac_edit)
            elif reac_edit[-10:] == '0:0.0:-1.0':   
                reac_edits_a.append(reac_edit)
            elif reac_edit[-9:] == '0:0.0:1.0':   
                reac_edits_a.append(reac_edit)

            elif reac_edit[-9:] == '0:0.0:2.0':    
                reac_edits_a.append(reac_edit)

            else:
                reac_edits_b.append(reac_edit)
        
  
        for reac_edit in reac_edits_a:
            if reac_edit[:3] == '0:0':  
                pass
            elif reac_edit[-7:] == '0.0:0.0':
                pass
            elif reac_edit[-10:] == '0:0.0:-1.0':
                edit_map_num_lis = reac_edit.split(':')[:2]
                attach_map_num_1 = [int(i) for i in edit_map_num_lis if int(i) in prod_map_num_lis] 
                lg_smiles = '-1'
                lg_map_lis.append((lg_smiles,attach_map_num_1))
            elif reac_edit[-9:] == '0:0.0:1.0':
                edit_map_num_lis = reac_edit.split(':')[:2]
                attach_map_num_1 = [int(i) for i in edit_map_num_lis if int(i) in prod_map_num_lis] 
                lg_smiles = '1'
                lg_map_lis.append((lg_smiles,attach_map_num_1))

            elif reac_edit[-9:] == '0:0.0:2.0':
                edit_map_num_lis = reac_edit.split(':')[:2]
                attach_map_num_1 = [int(i) for i in edit_map_num_lis if int(i) in prod_map_num_lis] 
                lg_smiles = '2'
                lg_map_lis.append((lg_smiles,attach_map_num_1))

                
        frag_1_map_num_lis = [i.GetAtomMapNum() for i in frag_mols_1.GetAtoms() if i.GetAtomMapNum() != 0]
        reac_frag_mol = apply_edits_to_mol_break(reac_mols_1 , reac_edits_b)
        reac_frag_mols = Chem.GetMolFrags(reac_frag_mol,asMols=True,sanitizeFrags = False)
        
        
        reac_edit_added = []
        for reac_frag_mol in reac_frag_mols[:]:

            reac_frag_map_num_lis = [i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms() if i.GetAtomMapNum() != 0]

            if set(reac_frag_map_num_lis) == set(frag_1_map_num_lis): 
                pass
            else:
                attach_map_num_1 = []           
                for reac_edit in reac_edits:    
                    if reac_edit in reac_edit_added:
                        continue
                    else:
                        pass

                    
                    b,e = int(reac_edit.split(':')[0]),int(reac_edit.split(':')[1]) 
                    if e in reac_frag_map_num_lis and b in frag_1_map_num_lis:

                        for atom in reac_frag_mol.GetAtoms(): 
                            if atom.GetAtomMapNum() == int(e):
                                atom.SetAtomMapNum(500+atom.GetAtomMapNum())
                                break  
                            else:
                                pass
                        reac_edit_added.append(reac_edit)
                        
                        

                        if len(attach_map_num_1) == 1:

                            if [str(attach_map_num_1[0]),str(atom.GetAtomMapNum()-500)] in [i.split(':')[:2] for i in reac_edits ]: #上一个合成子上的连接点和本离去基团的连接点配对
                                if atom.GetAtomMapNum() == max([i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms()]):
                                    attach_map_num_1 = [b] + attach_map_num_1
                                else:
                                    attach_map_num_1.append(b)
                            else: 
                                if atom.GetAtomMapNum() == max([i.GetAtomMapNum() for i in reac_frag_mol.GetAtoms()]):
                                    attach_map_num_1.append(b)
                                else:
                                    attach_map_num_1 = [b] + attach_map_num_1
                        elif len(attach_map_num_1) == 0:
                            attach_map_num_1.append(b)
                            

                    else:
                        pass
                        
                    if reac_frag_mol.GetAtomWithIdx(0).GetAtomicNum() == 1 and len(attach_map_num_1) == 1:  
                        break

                
                lg_smiles = Chem.MolToSmiles(reac_frag_mol,kekuleSmiles = True)  
                lg = Chem.MolFromSmiles(lg_smiles)
                Chem.Kekulize(lg)
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() >= 500:
                        atom.SetAtomMapNum(1)
                        pass
                    else:
                        atom.SetAtomMapNum(0)
                lg_smiles = Chem.MolToSmiles(lg,canonical = False,kekuleSmiles = True)             
                
                if attach_map_num_1 != []:
                    lg_map_lis.append((lg_smiles,attach_map_num_1))
                    
    return lg_map_lis



def get_chai_edit_mine(reac_mol, prod_mol):    
    reac_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()}
    prod_map_a = {atom.GetIdx(): atom.GetAtomMapNum() for atom in prod_mol.GetAtoms()}
    
    reac_mol_= copy.deepcopy(reac_mol)
    prod_mol_= copy.deepcopy(prod_mol)
    
    for atom in reac_mol_.GetAtoms():
        atom.SetAtomMapNum(0)
        
    for atom in prod_mol_.GetAtoms():
        atom.SetAtomMapNum(0)
    
    
    reac_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(reac_mol_)),includeUnassigned=True):
        reac_ChiralCenters.append((reac_map_a[ChiralCenters[0]],ChiralCenters[1]))

    prod_ChiralCenters = []
    for ChiralCenters in Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(prod_mol_)),includeUnassigned=True):
        prod_ChiralCenters.append((prod_map_a[ChiralCenters[0]],ChiralCenters[1]))
        
    dict_reac_ChiralCenters = dict(reac_ChiralCenters)
    dict_prod_ChiralCenters = dict(prod_ChiralCenters)
    
    
    chai_edits = []

    for amap_num,chiral in dict_prod_ChiralCenters.items():
        if amap_num in dict_reac_ChiralCenters.keys():
            if chiral != dict_reac_ChiralCenters[amap_num]:
                edit = f"{amap_num}:{0}:{0}:{dict_reac_ChiralCenters[amap_num]}"
                chai_edits.append(edit)
        else:
            pass
    
    for amap_num,chiral in dict_reac_ChiralCenters.items(): 
        if (amap_num not in dict_prod_ChiralCenters.keys()) and (amap_num in prod_map_a.values()) and chiral != '?':
            edit = f"{amap_num}:{0}:{0}:{chiral}"
            chai_edits.append(edit)
        
    return chai_edits



def get_original_chair_edit(p,b):
    b = copy.deepcopy(b)
    for atom in b.GetAtoms():
        atom.SetAtomMapNum(0)    
    b_dic = dict(Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(b)),includeUnassigned=True))
    
    temp_p = Chem.MolFromSmiles(p)
    for atom in temp_p.GetAtoms():
        atom.SetAtomMapNum(0)
    temp_dic = dict(Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(temp_p)),includeUnassigned=True))
    out = []
    for i,j in temp_dic.items():
        if i in b_dic:
            out.append('{}:0:0:{}'.format(i+1,j))
    return out




def apply_chirality_change(prod_mol,chai_edits):
    p_amap_idx =  {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}
    prod_mol = copy.deepcopy(prod_mol)
    for chai_edit in chai_edits:
        amap = int(chai_edit.split(':')[0])
        if chai_edit[-2:] == ':R':
            atom = prod_mol.GetAtomWithIdx(p_amap_idx[amap])
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            temp_mol_dic = get_chair_dict_without_atom_map(prod_mol)
            if temp_mol_dic[atom.GetIdx()] == 'R':
                pass
            else:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                
        elif chai_edit[-2:] == ':S':
            atom = prod_mol.GetAtomWithIdx(p_amap_idx[amap])
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            temp_mol_dic = get_chair_dict_without_atom_map(prod_mol)
            if temp_mol_dic[atom.GetIdx()] == 'S':
                pass
            else:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                temp_mol_dic = dict(Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(prod_mol)),includeUnassigned=True))
                
                
        elif chai_edit[-2:] == ':?':
            atom = prod_mol.GetAtomWithIdx(p_amap_idx[amap])
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            
    return  prod_mol




def get_chair_dict_without_atom_map(temp_p):
    temp_p = copy.deepcopy(temp_p)
    for atom in temp_p.GetAtoms():
        atom.SetAtomMapNum(0)
    temp_dic = dict(Chem.FindMolChiralCenters(Chem.MolFromMolBlock(Chem.MolToMolBlock(temp_p)),includeUnassigned=True))
    return temp_dic



def run_get_p_b_l(rxn_smi):
    try:
        r, p = rxn_smi.split(">>")
        
        if Chem.MolFromSmiles(p).GetNumAtoms() >= 150 or Chem.MolFromSmiles(r).GetNumAtoms() >= 150:
            print('error type 3')
            return 'error type 3'
        else:
            pass
         

        r,p = cano_smiles_map(r),cano_smiles_map(p) 

        reac_mol, prod_mol = align_kekule_pairs(r, p)  
        reac_mol = Chem.MolFromSmiles(Chem.MolToSmiles(reac_mol,kekuleSmiles = True),sanitize = False)


        reac_smiles_temp = Chem.MolToSmiles(reac_mol,kekuleSmiles = True)
        reac_mol_temp = Chem.MolFromSmiles(reac_smiles_temp)

        if reac_mol_temp != None and Chem.MolToSmiles(reac_mol_temp) == Chem.MolToSmiles(Chem.MolFromSmiles(r)):
            pass
        else:
            r_k = get_kekule_aligned_r(r,p)
            if count_kekule_d(r_k,p) == 0:
                reac_mol, prod_mol = Chem.MolFromSmiles(r_k),Chem.MolFromSmiles(p)
                Chem.Kekulize(reac_mol)
                Chem.Kekulize(prod_mol)
            else:
                reac_mol, prod_mol = Chem.MolFromSmiles(r_k),Chem.MolFromSmiles(p)
                Chem.Kekulize(reac_mol)
                Chem.Kekulize(prod_mol)



        core_edits_add = [i for i in core_edits if (float(i.split(':')[2]) == 0) and (float(i.split(':')[1]) != 0)]

        core_edits = [i for i in core_edits if i not in core_edits_add]


        edit_c = [i for i in core_edits if (float(i.split(':')[-1]) > 0)]
        edit_b = [i for i in core_edits if (float(i.split(':')[-1]) == 0)]

        chai_edits = get_chai_edit_mine(Chem.MolFromSmiles(r), Chem.MolFromSmiles(p))

        stereo_edits = get_stereo_edit_mine(Chem.MolFromSmiles(r), Chem.MolFromSmiles(p))

        

        charge_edits = get_charge_edit_mine(reac_mol, prod_mol,core_edits) 


        o_p_Chiral_dic = get_atom_map_chai_dic(Chem.MolFromSmiles(p)) 
        o_p_Stereo_dic = get_atom_map_stereo_dic(Chem.MolFromSmiles(p))


        frag_mol = apply_edits_to_mol_break(prod_mol,edit_b)
        frag_mol = apply_edits_to_mol_change(frag_mol,edit_c)

        frag_mol = apply_edits_to_mol_connect(frag_mol, core_edits_add)   
        frag_mol = remove_s_H(frag_mol)


        reac_mols = Chem.GetMolFrags(reac_mol,asMols=True,sanitizeFrags = False)
        frag_mols = Chem.GetMolFrags(frag_mol,asMols=True,sanitizeFrags = False)

        if len(reac_mols) != len(frag_mols):
            frag_mols = [frag_mol for frag_mol in frag_mols if Chem.MolToSmiles(frag_mol) != '[H]']
        else:
            pass

        if len(reac_mols) != len(frag_mols):
            frag_mols = [frag_mol]
        else:
            pass


        if len(reac_mols) == len(frag_mols):
            reac_mols, frag_mols = map_reac_and_frag(reac_mols,frag_mols)
        else:
            print('error type 0')


        lg_map_lis_temp = get_lg_map_lis(frag_mols[:],reac_mols[:],core_edits,prod_mol)

        lg_map_lis = []
        for lg, map_ in lg_map_lis_temp:
            lg, map_ = copy.deepcopy(lg),copy.deepcopy(map_)
            map_new = []
            if lg.count(':') > 1:
                lg = Chem.MolFromSmiles(lg)
                Chem.Kekulize(lg)
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() == 0:
                        map_new.append('*')
                    else:
                        map_new.append(map_.pop(0))

                lg_smiles = Chem.MolToSmiles(lg,kekuleSmiles = True)
                rank = list(Chem.CanonicalRankAtoms(lg, breakTies=False)) 
     

                map_new = sorted(map_new, key=lambda x: rank[map_new.index(x)])
                map_new = [i for i in map_new if  i != '*']

                lg_map_lis.append((lg_smiles,map_new))
            else:
                lg_map_lis.append((lg, map_ ))


        total_mol = frag_mol



        for lg_smile,map_nums in lg_map_lis[:]:

            if lg_smile not in ['-1.0','1.0','2.0']:

                lg = Chem.MolFromSmiles(lg_smile)

                total_mol_map_num_lis = [i.GetAtomMapNum() for i in total_mol.GetAtoms()]
                max_map = max(total_mol_map_num_lis)
                count = 1
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() == 1:
                        atom.SetAtomMapNum(max_map + count)
                        count += 1
                    else:
                        pass

                total_mol_map_num_lis = [i.GetAtomMapNum() for i in total_mol.GetAtoms()]
                max_map = max(total_mol_map_num_lis)

                for atom in lg.GetAtoms():   
                    if atom.GetAtomMapNum() == 0:
                        atom.SetAtomMapNum(max_map + count)
                        count += 1
                    else:
                        pass

                total_mol = Chem.CombineMols(total_mol,lg)

                amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in total_mol.GetAtoms()}
                new_mol = Chem.RWMol(total_mol)

                is_multi_bond = 0

                for idx in range(len(map_nums)):   
                    map_num = map_nums[idx]  
                    if lg_smile.count(':') == len(map_nums):  
                        lg_map = max_map + 1 + idx
                        atom = total_mol.GetAtomWithIdx(amap[lg_map])  
                        is_multi_bond = 0
                    else:
                        lg_map = max_map + 1 
                        atom = total_mol.GetAtomWithIdx(amap[lg_map]) 
                        is_multi_bond= 1  




                    if atom.GetSymbol()  == 'O' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == 0 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'S' and atom.GetTotalValence()  in [0,2,4] and atom.GetFormalCharge()  == 0 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'S' and atom.GetTotalValence()  ==1 and atom.GetFormalCharge()  == 1 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'P' and atom.GetTotalValence()  == 3 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 1  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == -1  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Se' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Si' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Mn' and atom.GetTotalValence()  == 5 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Cr' and atom.GetTotalValence()  == 4 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'O' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 1  and is_multi_bond == 0:
                        bond_float = 2.0

      
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 3.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 3.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == -1  and is_multi_bond == 0:
                        bond_float = 3.0




                    else:

                        bond_float = 1.0


                    new_mol.AddBond(amap[map_num],amap[lg_map],BOND_FLOAT_TO_TYPE[bond_float])
                total_mol = new_mol.GetMol()

            else:

                map_num = map_nums[0]

                amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in total_mol.GetAtoms()}
                atom = total_mol.GetAtomWithIdx(amap[map_num])
                atom.SetNumRadicalElectrons(0)
                atom.SetFormalCharge(int(atom.GetFormalCharge()+float(lg_smile)))


        total_mol = correct_mol_1(total_mol,is_nitrine_c = True)

        b = correct_mol(total_mol,keep_map = True)

        b_Chiral_dic = get_atom_map_chai_dic(b)
        b_Stereo_dic = get_atom_map_stereo_dic(b)

        dic_map_idx = dict([(i.GetAtomMapNum(),i.GetIdx()) for i in b.GetAtoms()])

        act = 0
        for b_map,Chiral in b_Chiral_dic.items():
            if b_map not in o_p_Chiral_dic.keys():
                pass
            elif b_map in o_p_Chiral_dic.keys() and b_Chiral_dic[b_map] != o_p_Chiral_dic[b_map] and b_map not in [int(i.split(':')[0]) for i in chai_edits]:

                act =1
                atom = b.GetAtomWithIdx(dic_map_idx[b_map])

                if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

        if act == 1:
            pass




        for b_map,Stereo in b_Stereo_dic.items(): 
            if b_map not in o_p_Stereo_dic.keys():
                pass
            elif b_map in o_p_Stereo_dic.keys() and Stereo != o_p_Stereo_dic[b_map] and b_map not in [tuple([int(i) for i in i.split(':')[:2]]) for i in stereo_edits]:
                bond = b.GetBondBetweenAtoms(dic_map_idx[b_map[0]],dic_map_idx[b_map[1]])

                bond.SetStereo(o_p_Stereo_dic[b_map])

        b = apply_charge_change(b,charge_edits)

        if chai_edits == []:
            o_chai_edits = get_original_chair_edit(p,b)


            b = apply_chirality_change(b,o_chai_edits)

        else:
            b = apply_chirality_change(b,chai_edits)

        
        
        b = Chem.MolFromSmiles(Chem.MolToSmiles(b,canonical = False)) 
        
        b = apply_stereo_change(b,stereo_edits)



        for atom in b.GetAtoms():
            atom.SetAtomMapNum(0)


        for bond in b.GetBonds():

            if bond.GetStereo() == Chem.rdchem.BondStereo.STEREONONE:

                bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY)
            else:
                pass


        pre_smiles = Chem.MolToSmiles(b)

        pre_smiles = pre_smiles.replace('[H]/C=C/','C=C').replace('[H]/C=C(\\','C=C(').replace('[H]/C=C(/','C=C(').replace('[MgH2]','[Mg]').replace('/C=N\\','C=C')

        pre_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(pre_smiles))

        reac_mol = Chem.MolFromSmiles(r)

        for atom in reac_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        reac_mol_smiles = Chem.MolToSmiles(reac_mol)

        reac_mol_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(reac_mol_smiles))
        
        


        if [float(i[-3:]) for i in core_edits_add] == []:
            max_add = 0
        elif max([float(i[-3:]) for i in core_edits_add]) == 1:
            max_add = 1
        else:
            max_add = 2
        
        charges = [int(i[-1]) for i in charge_edits] + [0]

        if pre_smiles == reac_mol_smiles and  len(core_edits_add) <= 1 and max_add <=1 and max(charges)<=1 and min(charges)>=-1:

            return ([p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis])
        else:
            print(pre_smiles,reac_mol_smiles,chai_edits,stereo_edits)
            return 'error type 1'

        
     
    
    except:
        print('error type 2')
        return 'error type 2'



def run_get_p_b_l_forward(rxn_smi):
    try:
        r, p = rxn_smi.split(">>")
        
        if Chem.MolFromSmiles(p).GetNumAtoms() >= 150 or Chem.MolFromSmiles(r).GetNumAtoms() >= 150:

            return 'error type 1'
        else:
            pass
         

        r,p = cano_smiles_map(r),cano_smiles_map(p) 
        reac_mol, prod_mol = align_kekule_pairs(r, p)   
        reac_mol = Chem.MolFromSmiles(Chem.MolToSmiles(reac_mol,kekuleSmiles = True),sanitize = False)


        reac_smiles_temp = Chem.MolToSmiles(reac_mol,kekuleSmiles = True)
        reac_mol_temp = Chem.MolFromSmiles(reac_smiles_temp)

        if reac_mol_temp != None and Chem.MolToSmiles(reac_mol_temp) == Chem.MolToSmiles(Chem.MolFromSmiles(r)):
            pass
        else:
            r_k = get_kekule_aligned_r(r,p)
            if count_kekule_d(r_k,p) == 0:
                reac_mol, prod_mol = Chem.MolFromSmiles(r_k),Chem.MolFromSmiles(p)
                Chem.Kekulize(reac_mol)
                Chem.Kekulize(prod_mol)
            else:
                reac_mol, prod_mol = Chem.MolFromSmiles(r_k),Chem.MolFromSmiles(p)
                Chem.Kekulize(reac_mol)
                Chem.Kekulize(prod_mol)




        core_edits= get_core_edit_mine(reac_mol,prod_mol) 
        core_edits_add = [i for i in core_edits if (float(i.split(':')[2]) == 0) and (float(i.split(':')[1]) != 0)]
        core_edits = [i for i in core_edits if i not in core_edits_add]


        edit_c = [i for i in core_edits if (float(i.split(':')[-1]) > 0)]
        edit_b = [i for i in core_edits if (float(i.split(':')[-1]) == 0)]

        chai_edits = get_chai_edit_mine(Chem.MolFromSmiles(r), Chem.MolFromSmiles(p))
        stereo_edits = get_stereo_edit_mine(Chem.MolFromSmiles(r), Chem.MolFromSmiles(p))
        charge_edits = get_charge_edit_mine(reac_mol, prod_mol,core_edits)  


        o_p_Chiral_dic = get_atom_map_chai_dic(Chem.MolFromSmiles(p))  
        o_p_Stereo_dic = get_atom_map_stereo_dic(Chem.MolFromSmiles(p))


        frag_mol = apply_edits_to_mol_break(prod_mol,edit_b)
        frag_mol = apply_edits_to_mol_change(frag_mol,edit_c)

        frag_mol = apply_edits_to_mol_connect(frag_mol, core_edits_add)  
        frag_mol = remove_s_H(frag_mol)
  

        reac_mols = Chem.GetMolFrags(reac_mol,asMols=True,sanitizeFrags = False)
        frag_mols = Chem.GetMolFrags(frag_mol,asMols=True,sanitizeFrags = False)

        if len(reac_mols) != len(frag_mols):
            frag_mols = [frag_mol for frag_mol in frag_mols if Chem.MolToSmiles(frag_mol) != '[H]']
        else:
            pass

        if len(reac_mols) != len(frag_mols):
            frag_mols = [frag_mol]
        else:
            pass


        if len(reac_mols) == len(frag_mols):
            reac_mols, frag_mols = map_reac_and_frag(reac_mols,frag_mols)
        else:

            pass



        lg_map_lis_temp = get_lg_map_lis(frag_mols[:],reac_mols[:],core_edits,prod_mol)

        lg_map_lis = []
        for lg, map_ in lg_map_lis_temp:
            lg, map_ = copy.deepcopy(lg),copy.deepcopy(map_)
            map_new = []
            if lg.count(':') > 1:
                lg = Chem.MolFromSmiles(lg)
                Chem.Kekulize(lg)
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() == 0:
                        map_new.append('*')
                    else:
                        map_new.append(map_.pop(0))

                lg_smiles = Chem.MolToSmiles(lg,kekuleSmiles = True)
                rank = list(Chem.CanonicalRankAtoms(lg, breakTies=False)) 
                map_new = sorted(map_new, key=lambda x: rank[map_new.index(x)])
                map_new = [i for i in map_new if  i != '*']

                lg_map_lis.append((lg_smiles,map_new))
            else:
                lg_map_lis.append((lg, map_ ))




        return ([p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis])

        
     
    
    except:
        return 'error type 2'



def run_get_p_b_l_backward(p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis):

        prod_mol = Chem.MolFromSmiles(p)

        core_edits = [i for i in core_edits if i not in core_edits_add]
        edit_c = [i for i in core_edits if (float(i.split(':')[-1]) > 0)]
        edit_b = [i for i in core_edits if (float(i.split(':')[-1]) == 0)]


        o_p_Chiral_dic = get_atom_map_chai_dic(Chem.MolFromSmiles(p))  #
        o_p_Stereo_dic = get_atom_map_stereo_dic(Chem.MolFromSmiles(p))


        frag_mol = apply_edits_to_mol_break(prod_mol,edit_b)
        frag_mol = apply_edits_to_mol_change(frag_mol,edit_c)

        frag_mol = apply_edits_to_mol_connect(frag_mol, core_edits_add)   
        frag_mol = remove_s_H(frag_mol)



        total_mol = frag_mol


        for lg_smile,map_nums in lg_map_lis[:]:

            if lg_smile not in ['-1','1','2']: 

                lg = Chem.MolFromSmiles(lg_smile)

                total_mol_map_num_lis = [i.GetAtomMapNum() for i in total_mol.GetAtoms()]
                max_map = max(total_mol_map_num_lis)
                count = 1
                for atom in lg.GetAtoms():
                    if atom.GetAtomMapNum() == 1:
                        atom.SetAtomMapNum(max_map + count)
                        count += 1
                    else:
                        pass

                total_mol_map_num_lis = [i.GetAtomMapNum() for i in total_mol.GetAtoms()]
                max_map = max(total_mol_map_num_lis)

                for atom in lg.GetAtoms():  
                    if atom.GetAtomMapNum() == 0:
                        atom.SetAtomMapNum(max_map + count)
                        count += 1
                    else:
                        pass

                total_mol = Chem.CombineMols(total_mol,lg)

                amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in total_mol.GetAtoms()}
                new_mol = Chem.RWMol(total_mol)

                is_multi_bond = 0

                for idx in range(len(map_nums)):   
                    map_num = map_nums[idx] 
                    if lg_smile.count(':') == len(map_nums):  
                        lg_map = max_map + 1 + idx
                        atom = total_mol.GetAtomWithIdx(amap[lg_map])  
                        is_multi_bond = 0
                    else:
                        lg_map = max_map + 1 
                        atom = total_mol.GetAtomWithIdx(amap[lg_map])  
                        is_multi_bond= 1  


                    if atom.GetSymbol()  == 'O' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == 0 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'S' and atom.GetTotalValence()  in [0,2,4] and atom.GetFormalCharge()  == 0 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'S' and atom.GetTotalValence()  ==1 and atom.GetFormalCharge()  == 1 and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'P' and atom.GetTotalValence()  == 3 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 1  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == -1  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Se' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Si' and atom.GetTotalValence()  == 2 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Mn' and atom.GetTotalValence()  == 5 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'Cr' and atom.GetTotalValence()  == 4 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 2.0
                    elif atom.GetSymbol()  == 'O' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 1  and is_multi_bond == 0:
                        bond_float = 2.0


                    elif atom.GetSymbol()  == 'N' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 3.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 1 and atom.GetFormalCharge()  == 0  and is_multi_bond == 0:
                        bond_float = 3.0
                    elif atom.GetSymbol()  == 'C' and atom.GetTotalValence()  == 0 and atom.GetFormalCharge()  == -1  and is_multi_bond == 0:
                        bond_float = 3.0
                    else:

                        bond_float = 1.0


                    new_mol.AddBond(amap[map_num],amap[lg_map],BOND_FLOAT_TO_TYPE[bond_float])
                total_mol = new_mol.GetMol()

            else:

                map_num = map_nums[0]

                amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in total_mol.GetAtoms()}
                atom = total_mol.GetAtomWithIdx(amap[map_num])
                atom.SetNumRadicalElectrons(0)
                atom.SetFormalCharge(int(atom.GetFormalCharge()+float(lg_smile)))


        total_mol = correct_mol_1(total_mol,is_nitrine_c = True)

        b = correct_mol(total_mol,keep_map = True)

        b_Chiral_dic = get_atom_map_chai_dic(b)
        b_Stereo_dic = get_atom_map_stereo_dic(b)

        dic_map_idx = dict([(i.GetAtomMapNum(),i.GetIdx()) for i in b.GetAtoms()])

        act = 0
        for b_map,Chiral in b_Chiral_dic.items():
            if b_map not in o_p_Chiral_dic.keys():
                pass
            elif b_map in o_p_Chiral_dic.keys() and b_Chiral_dic[b_map] != o_p_Chiral_dic[b_map] and b_map not in [int(i.split(':')[0]) for i in chai_edits]:
                act =1
                atom = b.GetAtomWithIdx(dic_map_idx[b_map])

                if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

        if act == 1:
            pass
 



        for b_map,Stereo in b_Stereo_dic.items(): 
            if b_map not in o_p_Stereo_dic.keys():
                pass
            elif b_map in o_p_Stereo_dic.keys() and Stereo != o_p_Stereo_dic[b_map] and b_map not in [tuple([int(i) for i in i.split(':')[:2]]) for i in stereo_edits]:
                bond = b.GetBondBetweenAtoms(dic_map_idx[b_map[0]],dic_map_idx[b_map[1]])

                bond.SetStereo(o_p_Stereo_dic[b_map])
 
        b = apply_charge_change(b,charge_edits)

        if chai_edits == []:
            o_chai_edits = get_original_chair_edit(p,b)


            b = apply_chirality_change(b,o_chai_edits)

        else:
            b = apply_chirality_change(b,chai_edits)

        
        b = Chem.MolFromSmiles(Chem.MolToSmiles(b,canonical = False))  
        b = apply_stereo_change(b,stereo_edits)



        for atom in b.GetAtoms():
            atom.SetAtomMapNum(0)


        for bond in b.GetBonds():

            if bond.GetStereo() == Chem.rdchem.BondStereo.STEREONONE:

                bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY)
            else:
                pass

        pre_smiles = Chem.MolToSmiles(b)
        pre_smiles = pre_smiles.replace('[H]/C=C/','C=C').replace('[H]/C=C(\\','C=C(').replace('[H]/C=C(/','C=C(').replace('[MgH2]','[Mg]').replace('/C=N\\','C=C')
        pre_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(pre_smiles))
        return pre_smiles
    
    
    
def run_get_p_b_l_check(rxn):
    try:
        p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis =  run_get_p_b_l_forward(rxn)
    except:
        return 'error type 3'
    
    try:
        pre_smiles = run_get_p_b_l_backward(p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis)  # 加个5
    except:
        return 'error type 5'       

    r = rxn.split('>>')[0]
    reac_mol = Chem.MolFromSmiles(r)
    for atom in reac_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    reac_mol_smiles = Chem.MolToSmiles(reac_mol)
    reac_mol_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(reac_mol_smiles))


    if [float(i[-3:]) for i in core_edits_add] == []:
        max_add = 0
    elif max([float(i[-3:]) for i in core_edits_add]) == 1:
        max_add = 1
    else:
        max_add = 2

    charges = [int(i[-1]) for i in charge_edits] + [0]

    if pre_smiles == reac_mol_smiles and  len(core_edits_add) <= 1 and max_add <=1 and max(charges)<=1 and min(charges)>=-1:
        return p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis

    else:
        return 'error type 4'
        
        

        
        

def get_atom_pair_bond_idx_dic(concise_smiles):
    mol_indigo = indigo.loadMolecule(concise_smiles)
    mol_block_indigo = mol_indigo.molfile()                    
    
    mol = Chem.MolFromSmiles(concise_smiles,sanitize = False)
    atom_num = len(mol.GetAtoms())
    bond_num = len(mol.GetBonds())

    mol_block_lis = mol_block_indigo.split('\n')
    bond_line_lis = mol_block_lis[4+atom_num:4+atom_num+bond_num]
    atom_pair_bond_idx_dic = {}
    
    
    count = 0
    for bond_line in bond_line_lis:
        s_atom = int(bond_line[:3])
        e_atom = int(bond_line[3:6])
        min_atom = min((s_atom,e_atom))
        max_atom = max((s_atom,e_atom))
        atom_pair_bond_idx_dic[(min_atom,max_atom)] = count
        count += 1
        
    return atom_pair_bond_idx_dic




def get_rm_token_lis(concise_smiles,detailed_smiles):
    detailed_smiles_length = len(detailed_smiles)
    idx = 0
    rm_token_lis = []
    for _ in range(len(detailed_smiles)):

        if detailed_smiles[idx] != concise_smiles[idx]:
            rm_token_lis.append(detailed_smiles[idx])
            detailed_smiles = detailed_smiles[:idx] + detailed_smiles[idx+1:]
        else:
            idx += 1
            rm_token_lis.append(' ')
    if detailed_smiles ==  concise_smiles and len(rm_token_lis) == detailed_smiles_length:
        return rm_token_lis
    else:
        print('error')
        pass
    
    
def get_bond_token_lis(detailed_smiles):
    bond_token_lis = []

    for i in range(len(detailed_smiles)):
        
        if detailed_smiles[i] in ['-','=','#',':','/','\\'] and detailed_smiles[i+1] != ']':
            bond_token_lis.append(detailed_smiles[i])
        else:
            bond_token_lis.append(' ')
            pass
            
    return bond_token_lis


def get_bond_token_idx_dic(bond_token_lis):
    bond_token_idx_dic = {}
    bond_idx = 0
    token_idx = 0
    for i in bond_token_lis:
        token_idx += 1
        if i != ' ':
            bond_idx += 1
        else:
            pass
        bond_token_idx_dic[bond_idx] = token_idx
    return bond_token_idx_dic


def rerank_special_bond(mol_block_indigo_lis,bond_idx):
    mol = Chem.MolFromMolBlock('\n'.join(mol_block_indigo_lis),removeHs = False)
    q = mol_block_indigo_lis[mol.GetNumAtoms()+ 4 +bond_idx][:3]
    h = mol_block_indigo_lis[mol.GetNumAtoms()+ 4 +bond_idx][3:6]
    mol_block_indigo_lis[mol.GetNumAtoms()+ 4 +bond_idx] =  h + q + mol_block_indigo_lis[mol.GetNumAtoms()+ 4 +bond_idx][6:]
    return  mol_block_indigo_lis

def get_caption_r(caption):
    words = re.findall(r'[{](.*?)[}]', caption)
    words = ['{' + i + '}' for i in words ]
    caption_r = caption
    count = 400
    for i in words:
        count += 1
        caption_r = caption_r.replace(i,'[{}Au]'.format(count),1)    
    
    return caption_r,words


def get_b_smiles_detailed_smiles(caption_r,smiles):
    b_smiles = caption_r
    
    b_smiles = b_smiles.replace('/','/-').replace('\\','\\-')  
    b_smiles = b_smiles.replace('-!','!').replace('-?','?')  
    
    mol_tmp = Chem.MolFromSmiles(smiles,sanitize = False)
    detailed_smiles = Chem.MolToSmiles(mol_tmp,canonical = False,allBondsExplicit = True)
    
    detailed_smiles = detailed_smiles.replace('/','/-').replace('\\','\\-') #
    
    for i in range(len(detailed_smiles)):
        if detailed_smiles[i] != b_smiles[i]:
            if b_smiles[i] in ['!','_',';','^','&','{','}','。','《','》']:
                pass
            else:
                b_smiles = b_smiles[:i] + detailed_smiles[i] + b_smiles[i:] 
        else:
            pass
        
    return b_smiles,detailed_smiles
        

def get_bond_dic(b_smiles,detailed_smiles):
    b_smiles = b_smiles.replace('-]',']')
    detailed_smiles = detailed_smiles.replace('-]',']')
    count = 0
    bond_dic = {}
    for i,j in zip(detailed_smiles,b_smiles):
        if i != j:
            bond_dic[count] = j

        if i in ['-','=','#',':']:
            count += 1
    return bond_dic


def get_t_smiles(e_smiles,o_smiles):
    e_smiles_r = e_smiles.replace('!','-').replace('_','-').replace(';','-').replace('^','-').replace('&','=').replace('{','=').replace('}','=').replace('。','=').replace('《','=').replace('》','=')
    mol_r = Chem.MolFromSmiles(e_smiles_r,sanitize = False)
    a = Chem.MolFromSmiles(o_smiles,sanitize = False)
    
    for atom in a.GetAtoms():
        atom.SetAtomMapNum(0)
        
    for atom in mol_r.GetAtoms():
        if atom.GetIsotope() != 0:
            a.GetAtomWithIdx(atom.GetIdx()).SetIsotope(atom.GetIsotope())
            
    t_smiles = Chem.MolToSmiles(a,canonical = False)
    return t_smiles




def get_b_smiles(p_b):

    o_smiles = p_b[0]
    core_edits = p_b[1]     
    chai_edits =  p_b[2]
    stereo_edits =  p_b[3]
    charge_edits =  p_b[4]
    core_edits_add =  p_b[5]
    atom_idx_mark_dic = {}   

    for edit in core_edits:
        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        new_b = edit.split(':')[3]
        if min([b,e]) == 0:
            atom_map = max([b,e])
            if new_b == '0.0':
                atom_idx_mark_dic[atom_map] = 9 
            else:
                pass



    for edit in chai_edits:

        edit_l = edit.split(':')
        if edit_l[3] == 'R': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 10
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 10 + atom_idx_mark_dic[int(edit_l[0])]
        elif edit_l[3] == 'S': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 20
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 20 + atom_idx_mark_dic[int(edit_l[0])]
        elif edit_l[3] == '?': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 30
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 30 + atom_idx_mark_dic[int(edit_l[0])]


    for edit in charge_edits: 

        edit_l = edit.split(':')
        if edit_l[3] == '1': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 200
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 200 + atom_idx_mark_dic[int(edit_l[0])]
                pass
            
        elif edit_l[3] == '0': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 400
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 400 + atom_idx_mark_dic[int(edit_l[0])]

            
        elif edit_l[3] == '-1': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 600
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 600 + atom_idx_mark_dic[int(edit_l[0])]

                
    
    for edit in core_edits_add:
        edit_l = edit.split(':')

        if int(edit_l[0]) not in atom_idx_mark_dic.keys():
            atom_idx_mark_dic[int(edit_l[0])] = 100
        else:
            atom_idx_mark_dic[int(edit_l[0])] = 100 + atom_idx_mark_dic[int(edit_l[0])]

            
        if int(edit_l[1]) not in atom_idx_mark_dic.keys():
            atom_idx_mark_dic[int(edit_l[1])] = 100
        else:
            atom_idx_mark_dic[int(edit_l[1])] = 100 + atom_idx_mark_dic[int(edit_l[1])]


                
                

    a = Chem.MolFromSmiles(o_smiles,sanitize = False)

    for atom in a.GetAtoms():
        if atom.GetAtomMapNum() in atom_idx_mark_dic.keys():
            atom_map = atom.GetAtomMapNum()
            atom.SetIsotope(atom_idx_mark_dic[atom_map])            
        else:
            pass
        atom.SetAtomMapNum(0)

    mol = copy.deepcopy(a)


    detailed_smiles = Chem.MolToSmiles(mol,canonical = False,allBondsExplicit = True,kekuleSmiles=True)  


    concise_smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True)                
    concise_smiles_no_chirality = Chem.MolToSmiles(mol,canonical = False,isomericSmiles = False,kekuleSmiles=True)      
    atom_pair_bond_idx_dic = get_atom_pair_bond_idx_dic(concise_smiles_no_chirality)      
    rm_token_lis = get_rm_token_lis(concise_smiles,detailed_smiles)         
    bond_token_lis = get_bond_token_lis(detailed_smiles)                     
    bond_token_idx_dic = get_bond_token_idx_dic(bond_token_lis)              


    bond_idx_mark_dic = {}
    for edit in core_edits:

        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        org_b = edit.split(':')[2]
        new_b = edit.split(':')[3]
        if min([b,e]) != 0:
            bond_idx = atom_pair_bond_idx_dic[min([b,e]),max([b,e])]
            if new_b == '0.0':
                mark = '!'
            elif new_b == '1.0':
                mark = '_'
            elif new_b == '2.0':
                mark = ';'
            elif new_b == '3.0':
                mark = '^'
            bond_idx_mark_dic[bond_idx] = mark
        else:
            pass

    for edit in stereo_edits:   

        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        new_b = edit.split(':')[3]
        if min([b,e]) != 0:
            bond_idx = atom_pair_bond_idx_dic[min([b,e]),max([b,e])]
            if bond_idx not in bond_idx_mark_dic.keys():  

                if new_b == 'a':
                    mark = '&'
                elif new_b == 'e':
                    mark = '{'
                elif new_b == 'z':
                    mark = '}'
                bond_idx_mark_dic[bond_idx] = mark
            else:
                bond_idx in bond_idx_mark_dic.keys()      
                if new_b == 'a':
                    mark = '。'
                elif new_b == 'e':
                    mark = '《'
                elif new_b == 'z':
                    mark = '》'
                bond_idx_mark_dic[bond_idx] = mark
        else:
            pass        



    for bond_idx,mark in bond_idx_mark_dic.items():
        token_idx = bond_token_idx_dic[bond_idx]
        rm_token_lis[token_idx] = mark

    new_smiles_lis = []
    for i in range(len(rm_token_lis)):
        if rm_token_lis[i] == ' ':
            new_smiles_lis.append(detailed_smiles[i])
        elif rm_token_lis[i][-1] in ['!','_',';','^','&','{','}','。','《','》']:
            new_smiles_lis.append(rm_token_lis[i])
        else:
            pass

    caption = ''.join(new_smiles_lis)
    out_b_smiles_lis.append(caption)



    caption_r = caption 



    t_smiles = get_t_smiles(caption_r,o_smiles)

    b_smiles,detailed_smiles = get_b_smiles_detailed_smiles(caption_r,t_smiles)


    bond_dic = get_bond_dic(b_smiles,detailed_smiles)


    atom_pair_bond_idx = {}
    for atom_pair,bond_idx in get_atom_pair_bond_idx_dic(o_smiles).items():
        atom_pair_bond_idx[bond_idx] = atom_pair



    mol = Chem.MolFromSmiles(t_smiles)
    Chem.Kekulize(mol)
    core_edits_ = []
    chai_edits_ = []
    stereo_edits_ =  []
    charge_edits_ =  []
    core_edits_add_ = []

    for bond_idx,mark in bond_dic.items():    
        b,e = atom_pair_bond_idx[bond_idx]
        o_bond = mol.GetBondBetweenAtoms(b-1,e-1).GetBondTypeAsDouble()
        if mark == '!':
            n_bond = '0.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == '_':
            n_bond = '1.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == ';':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == '^':
            n_bond = '3.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))

        elif mark == '&':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'a'))
        elif mark == '{':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'e'))      
        elif mark == '}':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'z'))   


        elif mark == '。': 
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'a'))   
        elif mark == '《':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'e'))
        elif mark == '》':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'z'))


    
    core_edits_add_atom_lis = []
    
    for atom in mol.GetAtoms():   
        Isotope = atom.GetIsotope()
        g_w = Isotope % 10
        s_w = Isotope % 100 //10
        b_w = Isotope // 100
        
        if g_w == 9:
            core_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'1.0','0.0'))  
        else:
            pass
            
            
        if s_w == 1:   
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','R'))
        elif s_w == 2:
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','S'))
        elif s_w == 3:
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','?'))


        if b_w == 2 or b_w == 3:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',1))
        elif b_w == 4 or b_w == 5:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',0))
        elif b_w == 6 or b_w == 7:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',-1))    

        
        if b_w % 2 == 1:
            core_edits_add_atom_lis.append(atom.GetIdx()+1)
            

    if core_edits_add_atom_lis != []:
        core_edits_add_.append('{}:{}:{}:{}'.format(core_edits_add_atom_lis[0],core_edits_add_atom_lis[1],'0.0','1.0'))
    else:
        pass

    if sorted(core_edits_) != sorted(core_edits) or sorted(chai_edits_) != sorted(chai_edits) or sorted(stereo_edits_) != sorted(stereo_edits) or sorted(charge_edits_) != sorted(charge_edits) or sorted(core_edits_add_) != sorted(core_edits_add):
        print(core_edits_,core_edits)
        print(chai_edits_,chai_edits)
        print(core_edits_add_,core_edits_add)
        return 'error'
    else:
        return caption
        pass
    
    
def get_b_smiles_forward(p_b):
    o_smiles = p_b[0]
    core_edits = p_b[1]     
    chai_edits =  p_b[2]
    stereo_edits =  p_b[3]
    charge_edits =  p_b[4]
    core_edits_add =  p_b[5]
    atom_idx_mark_dic = {}   


    for edit in core_edits:
        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        new_b = edit.split(':')[3]
        if min([b,e]) == 0:
            atom_map = max([b,e])
            if new_b == '0.0':
                atom_idx_mark_dic[atom_map] = 9 
            else:
                pass


    for edit in chai_edits:

        edit_l = edit.split(':')
        if edit_l[3] == 'R': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 10
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 10 + atom_idx_mark_dic[int(edit_l[0])]
        elif edit_l[3] == 'S': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 20
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 20 + atom_idx_mark_dic[int(edit_l[0])]
        elif edit_l[3] == '?': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 30
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 30 + atom_idx_mark_dic[int(edit_l[0])]


    for edit in charge_edits:  

        edit_l = edit.split(':')
        if edit_l[3] == '1': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 200
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 200 + atom_idx_mark_dic[int(edit_l[0])]
                pass
            
        elif edit_l[3] == '0': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 400
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 400 + atom_idx_mark_dic[int(edit_l[0])]

            
        elif edit_l[3] == '-1': 
            if int(edit_l[0]) not in atom_idx_mark_dic.keys():
                atom_idx_mark_dic[int(edit_l[0])] = 600
            else:
                atom_idx_mark_dic[int(edit_l[0])] = 600 + atom_idx_mark_dic[int(edit_l[0])]

    
    for edit in core_edits_add:
        edit_l = edit.split(':')

        if int(edit_l[0]) not in atom_idx_mark_dic.keys():
            atom_idx_mark_dic[int(edit_l[0])] = 100
        else:
            atom_idx_mark_dic[int(edit_l[0])] = 100 + atom_idx_mark_dic[int(edit_l[0])]

            
        if int(edit_l[1]) not in atom_idx_mark_dic.keys():
            atom_idx_mark_dic[int(edit_l[1])] = 100
        else:
            atom_idx_mark_dic[int(edit_l[1])] = 100 + atom_idx_mark_dic[int(edit_l[1])]
            

    a = Chem.MolFromSmiles(o_smiles,sanitize = False)

    for atom in a.GetAtoms():
        if atom.GetAtomMapNum() in atom_idx_mark_dic.keys():
            atom_map = atom.GetAtomMapNum()
            atom.SetIsotope(atom_idx_mark_dic[atom_map])            
        else:
            pass
        atom.SetAtomMapNum(0)

    mol = copy.deepcopy(a)


    detailed_smiles = Chem.MolToSmiles(mol,canonical = False,allBondsExplicit = True,kekuleSmiles=True)   


    concise_smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True)                
    concise_smiles_no_chirality = Chem.MolToSmiles(mol,canonical = False,isomericSmiles = False,kekuleSmiles=True)      
    atom_pair_bond_idx_dic = get_atom_pair_bond_idx_dic(concise_smiles_no_chirality)      
    rm_token_lis = get_rm_token_lis(concise_smiles,detailed_smiles)          
    bond_token_lis = get_bond_token_lis(detailed_smiles)                    
    bond_token_idx_dic = get_bond_token_idx_dic(bond_token_lis)              


    bond_idx_mark_dic = {}
    for edit in core_edits:

        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        org_b = edit.split(':')[2]
        new_b = edit.split(':')[3]
        if min([b,e]) != 0:
            bond_idx = atom_pair_bond_idx_dic[min([b,e]),max([b,e])]
            if new_b == '0.0':
                mark = '!'
            elif new_b == '1.0':
                mark = '_'
            elif new_b == '2.0':
                mark = ';'
            elif new_b == '3.0':
                mark = '^'
            bond_idx_mark_dic[bond_idx] = mark
        else:
            pass

    for edit in stereo_edits:  

        b = int(edit.split(':')[0])
        e = int(edit.split(':')[1])
        new_b = edit.split(':')[3]
        if min([b,e]) != 0:
            bond_idx = atom_pair_bond_idx_dic[min([b,e]),max([b,e])]
            if bond_idx not in bond_idx_mark_dic.keys():  

                if new_b == 'a':
                    mark = '&'
                elif new_b == 'e':
                    mark = '{'
                elif new_b == 'z':
                    mark = '}'
                bond_idx_mark_dic[bond_idx] = mark
            else:
                bond_idx in bond_idx_mark_dic.keys()    
                if new_b == 'a':
                    mark = '。'
                elif new_b == 'e':
                    mark = '《'
                elif new_b == 'z':
                    mark = '》'
                bond_idx_mark_dic[bond_idx] = mark
        else:
            pass        


    for bond_idx,mark in bond_idx_mark_dic.items():
        token_idx = bond_token_idx_dic[bond_idx]
        rm_token_lis[token_idx] = mark

    new_smiles_lis = []
    for i in range(len(rm_token_lis)):
        if rm_token_lis[i] == ' ':
            new_smiles_lis.append(detailed_smiles[i])
        elif rm_token_lis[i][-1] in ['!','_',';','^','&','{','}','。','《','》']:
            new_smiles_lis.append(rm_token_lis[i])
        else:
            pass
        
    return ''.join(new_smiles_lis)   


def get_b_smiles_backward(caption_r,o_smiles):
    
    t_smiles = get_t_smiles(caption_r,o_smiles)
    b_smiles,detailed_smiles = get_b_smiles_detailed_smiles(caption_r,t_smiles)
    bond_dic = get_bond_dic(b_smiles,detailed_smiles)


    atom_pair_bond_idx = {}
    for atom_pair,bond_idx in get_atom_pair_bond_idx_dic(o_smiles).items():
        atom_pair_bond_idx[bond_idx] = atom_pair



    mol = Chem.MolFromSmiles(t_smiles)
    Chem.Kekulize(mol)
    core_edits_ = []
    chai_edits_ = []
    stereo_edits_ =  []
    charge_edits_ =  []
    core_edits_add_ = []

    for bond_idx,mark in bond_dic.items():    
        b,e = atom_pair_bond_idx[bond_idx]
        o_bond = mol.GetBondBetweenAtoms(b-1,e-1).GetBondTypeAsDouble()
        if mark == '!':
            n_bond = '0.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == '_':
            n_bond = '1.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == ';':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
        elif mark == '^':
            n_bond = '3.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))

        elif mark == '&':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'a'))
        elif mark == '{':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'e'))      
        elif mark == '}':
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'z'))   


        elif mark == '。':  
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'a'))   #any
        elif mark == '《':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'e'))
        elif mark == '》':
            n_bond = '2.0'
            core_edits_.append('{}:{}:{}:{}'.format(b,e,o_bond,n_bond))
            stereo_edits_.append('{}:{}:{}:{}'.format(b,e,0,'z'))

    
    core_edits_add_atom_lis = []
    
    for atom in mol.GetAtoms(): 
        Isotope = atom.GetIsotope()
        g_w = Isotope % 10
        s_w = Isotope % 100 //10
        b_w = Isotope // 100
        
        if g_w == 9:
            core_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'1.0','0.0'))  
        else:
            pass
            
            
        if s_w == 1:   
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','R'))
        elif s_w == 2:
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','S'))
        elif s_w == 3:
            chai_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0','?'))


        if b_w == 2 or b_w == 3:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',1))
        elif b_w == 4 or b_w == 5:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',0))
        elif b_w == 6 or b_w == 7:
            charge_edits_.append('{}:{}:{}:{}'.format(atom.GetIdx()+1,0,'0',-1))    

            
        if b_w % 2 == 1:
            core_edits_add_atom_lis.append(atom.GetIdx()+1)

    if core_edits_add_atom_lis != []:
        core_edits_add_.append('{}:{}:{}:{}'.format(core_edits_add_atom_lis[0],core_edits_add_atom_lis[1],'0.0','1.0'))
    else:
        pass

    
    return core_edits_,chai_edits_,stereo_edits_,charge_edits_,core_edits_add_



def get_b_smiles_check(p_b):
    p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis = p_b
    b_smiles = get_b_smiles_forward(p_b)
    core_edits_,chai_edits_,stereo_edits_,charge_edits_,core_edits_add_ = get_b_smiles_backward(b_smiles,p_b[0])
    if sorted(core_edits_) != sorted(core_edits) or sorted(chai_edits_) != sorted(chai_edits) or sorted(stereo_edits_) != sorted(stereo_edits) or sorted(charge_edits_) != sorted(charge_edits) or sorted(core_edits_add_) != sorted(core_edits_add):
        print(core_edits_,core_edits)
        print(chai_edits_,chai_edits)
        print(core_edits_add_,core_edits_add)
        return 'error'
    else:
        return b_smiles
    
    
import re

def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString


def cano_smiles_map(smiles):
    atom_map_lis = []
    mol = Chem.MolFromSmiles(smiles,sanitize = False)
    for atom in mol.GetAtoms():
        atom_map_lis.append(atom.GetAtomMapNum())
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True)
    mol = Chem.MolFromSmiles(smiles,sanitize = False)    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom_map_lis[atom.GetIdx()])
    smiles = Chem.MolToSmiles(mol,canonical = False,kekuleSmiles=True) 
    return smiles


def get_lg_forward(core_edits,lg_map):
    
    attach_idx = []
    for core_edit in core_edits:
        core_edit = core_edit.split(':')
        if float(core_edit[2])-float(core_edit[3]) > 0:
            attach_idx.append(int(core_edit[0]))
            attach_idx.append(int(core_edit[1]))
        
    attach_idx = sorted(list(set(attach_idx)))
    attach_idx = [i for i in attach_idx if i != 0]
    lg_lis = [()]*len(attach_idx)

    for lg,map_lis in lg_map:

        if len(map_lis) == 1:
            map_ = map_lis[0]
            id_ = attach_idx.index(map_)
            lg_lis[id_] = tuple(list(lg_lis[id_]) +[lg])
            
        elif len(map_lis) != 1 and len(set(map_lis)) == 1:
            map_ = map_lis[0]
            id_ = attach_idx.index(map_)
            lg_lis[id_] = tuple(list(lg_lis[id_]) +[lg])
        elif len(map_lis) != 1 and len(set(map_lis)) != 1 and lg.count(':') == 1:
            for map_ in map_lis:
                id_ = attach_idx.index(map_)
                lg_lis[id_] = tuple(list(lg_lis[id_]) +[lg + "*"])
        elif len(map_lis) != 1 and len(set(map_lis)) != 1 and lg.count(':') == 2:

            if map_lis[0]<map_lis[1]:
                lg = replacenth(lg, ':1',':2',2)
            else:
                lg = lg.replace(':1',':2',1)
            for map_ in map_lis:
                id_ = attach_idx.index(map_)
                lg_lis[id_] = tuple(list(lg_lis[id_]) +[lg + "*"])
        else:
            print('error')

    return [tuple(sorted(i)) for i in lg_lis]



def get_lg_backward(core_edits_,lg_lis):

    attach_idx = []
    for core_edit in core_edits_:
        core_edit = core_edit.split(':')
        if float(core_edit[2])-float(core_edit[3]) > 0:
            attach_idx.append(int(core_edit[0]))
            attach_idx.append(int(core_edit[1]))
            
    attach_idx = [i for i in attach_idx if i != 0]
    attach_idx = sorted(list(set(attach_idx)))
    
    lg_map_new = []
    for id_,lg_ in zip(attach_idx,lg_lis):
        for lg in list(lg_):
            if lg.count(':') > 1:
            
                lg_map_new.append((lg,[id_]*lg.count(':')))
            else:

                lg_map_new.append((lg,[id_]))

    
    dic_t = {}
    for i,j in lg_map_new:  
        if '*' in i:
            dic_t.setdefault(i,[]).append(j[0])
        else:
            pass
    
    
    lg_map_new_k =[]
    for i,j in lg_map_new:
        if '*' not in i:
            lg_map_new_k.append((i,j)) 
            
        else:
            pass


    for i,j in dic_t.items():
        if ':2' not in i:
            lg_map_new_k.append((i.replace('*',''),j))
        elif i.index(':1') <= i.index(':2'):
            lg_map_new_k.append((i.replace('*','').replace(':2',':1'),j))
        else:
            j.reverse()
            lg_map_new_k.append((i.replace('*','').replace(':2',':1'),j))


    lg_map_new = lg_map_new_k
    return lg_map_new
    

    
    
dic_str_to_num = {}
for l in range(4,0,-1):
    for a,i in zip([0,200,400,600,100,300,500,700],['','α','β','γ','δ','αδ','βδ','γδ']):
        for b,j in zip([0,10,20,30],['','r','s','?']):
            for c,k in zip([0,9],['','~']):
                if len(k+j+i) == l:
                    dic_str_to_num[k+j+i] = str(a+b+c)
                    
                    
dic_num_to_str = {}
for l in range(3,0,-1):
    for a,i in zip([0,200,400,600,100,300,500,700],['','α','β','γ','δ','αδ','βδ','γδ']):
        for b,j in zip([0,10,20,30],['','r','s','?']):
            for c,k in zip([0,9],['','~']):
                if len(str(a+b+c)) == l and len(k+j+i) != 0:
                    dic_num_to_str[str(a+b+c)] = k+j+i
                    
            
            
def iso_to_symbo(txt,dic_num_to_str):

    for i,j in dic_num_to_str.items():
        i = '[' + i
        j = '[' + j
        txt = txt.replace(i,j)
    txt = txt.replace('。',';&').replace('》',';}').replace('《',';{')
    return txt

def symbo_to_iso(txt,dic_str_to_num):

    for i,j in dic_str_to_num.items():
        i = '[' + i
        j = '[' + j
        txt = txt.replace(i,j)
    txt = txt.replace(';&','。').replace(';}','》').replace(';{','《')
    return txt



def merge_smiles_only(text):


    text = symbo_to_iso(text,dic_str_to_num)
    o_smiles = text.split('>>>')[0]
    b_smiles = text.split('>>>')[1].split('<')[0]
    
    lg_lis = []
    for i in  re.findall(r"[<](.*?)[>]", text):
        if i == '':
            lg_lis.append(tuple())
        else:
            lg_lis.append(tuple(i.split(',')))
            
    core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add = get_b_smiles_backward(b_smiles,o_smiles)
    lg_map_lis = get_lg_backward(core_edits,lg_lis)
    
    p = Chem.MolFromSmiles(o_smiles,sanitize = False)
    for atom in p.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    p = Chem.MolToSmiles(p)
    
    pre_smiles = run_get_p_b_l_backward(p,core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis)
    
    return pre_smiles



def merge_smiles(text):
    try:
        return merge_smiles_only(text)
    except:
        return ""
    
    
def get_e_smiles(rxn):

    p_b = run_get_p_b_l_forward(rxn)
    b_smiles = get_b_smiles_check(p_b)
    lg_lis = get_lg_forward(p_b[1],p_b[6])

    k = p_b
    b = b_smiles
    c = lg_lis
    a = Chem.MolFromSmiles(k[0],sanitize = False)

    for atom in a.GetAtoms():
        atom.SetAtomMapNum(0)
    a = Chem.MolToSmiles(a,canonical = False)

    str_ = ''
    for i in c:
        str_ = str_ +  '<{}>'.format(','.join(i))
    txt = a +'>>>'+ b+str_

    return iso_to_symbo(txt,dic_num_to_str)

def get_e_smiles_with_check(rxn):

    p_b = run_get_p_b_l_check(rxn)
    b_smiles = get_b_smiles_check(p_b)
    lg_lis = get_lg_forward(p_b[1],p_b[6])

    k = p_b
    b = b_smiles
    c = lg_lis
    a = Chem.MolFromSmiles(k[0],sanitize = False)

    for atom in a.GetAtoms():
        atom.SetAtomMapNum(0)
    a = Chem.MolToSmiles(a,canonical = False)

    str_ = ''
    for i in c:
        str_ = str_ +  '<{}>'.format(','.join(i))
    txt = a +'>>>'+ b+str_

    return iso_to_symbo(txt,dic_num_to_str)

def get_edit_from_e_smiles(text):
    text = symbo_to_iso(text,dic_str_to_num)
    o_smiles = text.split('>>>')[0]
    b_smiles = text.split('>>>')[1].split('<')[0]
    
    lg_lis = []
    for i in  re.findall(r"[<](.*?)[>]", text):
        if i == '':
            lg_lis.append(tuple())
        else:
            lg_lis.append(tuple(i.split(',')))
            
    core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add = get_b_smiles_backward(b_smiles,o_smiles)
    lg_map_lis = get_lg_backward(core_edits,lg_lis)
    
    return core_edits,chai_edits,stereo_edits,charge_edits,core_edits_add,lg_map_lis