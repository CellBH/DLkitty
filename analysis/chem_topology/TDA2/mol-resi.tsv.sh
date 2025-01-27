#!/usr/bin/env zsh
cd $0:h
mlr -t --from mol.tsv label resn,x,y,z + put '$resi = NR' > mol-resi.tsv
