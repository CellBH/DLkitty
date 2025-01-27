#!/usr/bin/env zsh
cd $0:h
conda create -n kittycat numpy pandas rdkit
conda activate kittycat
pip install leidenalg
