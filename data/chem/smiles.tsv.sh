#!/usr/bin/env zsh
cd $0:h
mlr --c2t --from `git root`/data/raw/Sabio-RK_kcats_full.csv.gz rename SubstrateSMILES,SMILES +\
    cut -f Substrate,SMILES |
    sed 's/Union{[^}]*}//g' | sed 's/\["//g' | sed 's/"\]//g' | tr -d '"' |
    ../src/table_unjag2.sh 1 2 $'\t' ', ' |
    grep -v missing |
    mlr -t uniq -a + filter '$SMILES != ""' + put '$strlen = strlen($SMILES)' +\
    sort -n strlen -f SMILES > smiles.tsv

