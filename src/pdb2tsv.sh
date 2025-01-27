#!/usr/bin/env zsh
# Tested on AlphaFold2 model v4 mmCIF files which breaks the standard listed on
# https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
# USE: pdb2tsv.sh INFILE.pdb > OUTFILE.tsv

# first get the column names for atoms
grep '^_atom_site.' $1 | sed 's/_atom_site\.//' | tr '\n' '\t' | sed 's/\t$/\n/'
# get the atoms
grep '^ATOM' $1 | sed 's/  */\t/g' | sed 's/\t$//'


