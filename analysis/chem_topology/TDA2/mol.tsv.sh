#!/usr/bin/env zsh
cd $0:h
top=20
mlr -t --from `git root`/data/chem/smiles.tsv cut -f SMILES + tail -n $top + head -n1 | sed 1d | ./mol.tsv.py > mol.tsv
