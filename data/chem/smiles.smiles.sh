#!/usr/bin/env zsh
cd $0:h
mlr -t --from smiles.tsv cut -f SMILES | sed 1d > smiles.smiles
