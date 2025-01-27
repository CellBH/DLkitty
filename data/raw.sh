#!/usr/bin/env zsh
cd $0:h
mkdir -p raw/wildtypes/

gunzip -c ./wildtypes.tsv.gz | while read acc seq; do
    echo $acc
    if [ ! -f ./raw/wildtypes/$acc.tsv.gz ]; then
        {
            curl -sS "https://alphafold.ebi.ac.uk/files/AF-$acc-F1-model_v4.cif" > $acc.cif
            `git root`/src/pdb2tsv.sh $acc.cif | gzip > ./raw/wildtypes/$acc.tsv.gz && rm $acc.cif
        } &
    fi
done

# Clean up empty files and files that are empty after gunzip
# find . -size 0 | xargs rm
# find . -size 20c | xargs rm

