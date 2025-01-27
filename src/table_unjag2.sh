#!/usr/bin/env zsh
# unjag two paired columns from a table.
# example unjagging column 3 and 4 with delimiter tab and secondary delimiter semicolon:
# table_unjag2.sh 3 4 $'\t' ';' < infile.tsv > outfile.tsv
# assumes n == m
awk -F$3 'BEGIN{OFS="'$3'"}{
    n=split($'$1', a, "'$4'");
    m=split($'$2', b, "'$4'");
    for (i=1; i<=n; i++) {
        $'$1' = a[i];
        $'$2' = b[i];
        print
    }
}'
