#!/usr/bin/env zsh
cd $0:h
`git root`/src/PH2plotly.jl -L -o fig/ ./mol.h5
