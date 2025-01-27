#!/usr/bin/env zsh
cd $0:h
# make a command "git root" that gives the root folder of the repo.
git config alias.root 'rev-parse --show-toplevel'
./data/install.sh
