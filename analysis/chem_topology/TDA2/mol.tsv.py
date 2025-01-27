#!/usr/bin/env python3
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# long smiles selected from the raw/*.csv.gz sabio data
with sys.stdin as io:
    smiles = io.read().strip()

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
for i in range(10):
    if mol.GetNumConformers() > 0:
        break
    sys.stderr.write(f"Retrying making 3D attempt {i+2}\n")
    AllChem.EmbedMolecule(mol)

if mol.GetNumConformers() == 0:
    sys.stderr.write(f"Failed.\n")
    exit(1)

AllChem.UFFOptimizeMolecule(mol)

Draw.MolToFile(mol, "mol.svg")

for i, atom in enumerate(mol.GetAtoms()):
    pos = mol.GetConformer().GetAtomPosition(i)
    print(atom.GetSymbol(), pos.x, pos.y, pos.z, sep='\t')

