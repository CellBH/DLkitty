#!/usr/bin/env zsh
cd $0:h
mlr --c2t --from ./raw/Sabio-RK_kcats_full.csv.gz filter '$EnzymeType !=~ "mutant"' +\
    uniq -f UniProtID,ProteinSequences | sed 's/Union{Missing, String}//g' | tr -d '"[] ' |
    `git root`/src/table_unjag2.sh 1 2 $'\t' ',' | grep -v missing |
    sed 1d | sort -u | gzip > wildtypes.tsv.gz

