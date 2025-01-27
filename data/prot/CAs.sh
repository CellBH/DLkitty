#!/usr/bin/env zsh
cd $0:h
mkdir -p CAs/

for filename in ./raw/wildtypes/*.tsv.gz; do
    echo $filename
    mlr -t --from $filename rename -r '.*_x$,x,.*_y$,y,.*_z$,z,pdbx_sifts_xref_db_num,resi,pdbx_sifts_xref_db_res,resn,B_iso_or_equiv,pLDDT' +\
        filter '$auth_atom_id == "CA"' + cut -f 'x,y,z,resi,resn,pLDDT' | gzip > ./CAs/$filename:t
done

