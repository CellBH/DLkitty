This directory contains Python scripts and Jupyter notebooks to download and process kcat data from SABIO-RK and Brenda databases.
Files include:

* `Brenda/construct_dataset.ipynb` - extract kcats from Brenda and process the dataset
* `Sabio-RK/download.py` - dowload Sabio-RK dataset
* `Sabio-RK/construct_dataset.ipynb` - extract kcats from Sabio-RK and process the dataset
* `pubchem_synonyms.py` - retrieve substrate synonyms from PubChem
* `pubchem_SMILES.py` - retrieve SMILES info from PubChem
* `get_UniProtIDs_mp.py` - retrieve UniProtID sequences by querying UniProtKB
* `get_UniProtKB_sequences_mp.py` - retrieve UniProt sequences
* `merge_databases.ipynb` - remove kcat duplicates and merge the two databases
* `add_SMILES_UniProt.ipynb` - add SMILES and UniProt sequences
* `kcat_data_Julia_reformat.ipynb` - save data in Julia-friendly format

Following the repo structure, the data is saved in `data/databases/` (not uploaded on github but can be downloaded following [this link](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/augustinas_sukys_unimelb_edu_au/El5r5PN8jAlOiWh0Qa0rwXgBIYL8YXRDoBSPfA8EXUpueA)).