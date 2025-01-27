#!/usr/bin/env python3
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdFreeSASA
from rdkit.Chem import Draw

# long smiles selected from the raw/*.csv.gz sabio data
smiles = "C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)OP(=O)(O)O)O)O)O)C(=O)N"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
mol.GetNumConformers()
AllChem.UFFOptimizeMolecule(mol)

Draw.MolToFile(mol, "mol.svg")

for i, atom in enumerate(mol.GetAtoms()):
    pos = mol.GetConformer().GetAtomPosition(i)
    print(atom.GetSymbol(), pos.x, pos.y, pos.z, sep='\t')

