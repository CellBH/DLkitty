import numpy as np
import requests
import time
import csv
import tqdm

# NOTE:
# PubChem throttles requests (no more than five requests per second, otherwise leads to 503 error and blocks requests)
# Ideally would follow this https://medium.com/towards-data-science/responsible-concurrent-data-retrieval-80bf7911ca06

def get_SMILES(name, flog):

    n_retry = 5
    errors_str = name+':\n'
    i = 0
    while i < n_retry:
        i += 1
        t0 = time.time()
        try:
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
            req = requests.get(url)
            if req.status_code == 200:
                smiles = req.content.splitlines()[0].decode()
                return smiles
            elif req.status_code == 404:
                return None
            else:
                errors_str += 'Request error code: '+str(req.status_code)+'\n'
        except Exception as e:
            errors_str += str(e)+'\n'
        t1 = time.time()
        # Keep PubChem happy
        if (t1-t0) < 1:
            time.sleep(t1-t0)

    if i == n_retry:
        flog.write(errors_str)

    return None

if __name__ == '__main__':
    fdir_data = '../../data/databases/'
    fname = fdir_data+"log_download_PubChem_SMILES.out"
    flog = open(fname, 'w')
    all_substrates = np.load(fdir_data+'unique_substrates.npy')
    start = time.time()
    substrate_smiles_map = {}

    for substrate in tqdm.tqdm(all_substrates):
        t0 = time.time()
        substrate_name = get_SMILES(substrate, flog)
        substrate_smiles_map[substrate] = substrate_name
        t1 = time.time()
        if (t1-t0) < 1:
            time.sleep(t1-t0)

    end = time.time()
    print("Time: ", end - start)
    flog.close()

    with open(fdir_data+'substrate_SMILES_map_FINAL.csv','w') as f:
        w = csv.writer(f)
        w.writerows(substrate_smiles_map.items())