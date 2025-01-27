#!/usr/bin/env zsh
cd $0:h
mkdir -p PH/
`git root`/src/tsv2PH.jl ./CAs/*.tsv.gz PH/
