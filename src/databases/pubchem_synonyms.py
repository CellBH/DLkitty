import numpy as np
import requests 
import time
import csv
import tqdm

# NOTE: 
# PubChem throttles requests (no more than five requests per second, otherwise leads to 503 error and blocks requests) 
# Ideally would follow this https://medium.com/towards-data-science/responsible-concurrent-data-retrieval-80bf7911ca06

def get_substrate_synonym(name, flog):
    
    n_retry = 5
    errors_str = name+':\n'
    i = 0
    while i < n_retry:
        i += 1
        t0 = time.time()
        try:
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/synonyms/TXT' % name
            req = requests.get(url)
            if req.status_code == 200:
                substrate_name = req.content.splitlines()[0].decode()
                if (len(substrate_name) > 8) and (substrate_name[0:7] == "SCHEMBL"):
                    # unclear nomenclature otherwise
                    substrate_name = req.content.splitlines()[1].decode()
                return substrate_name
            elif req.status_code == 404:
                if ' ' in name:
                    name = name.replace(' ', '-')
                else:
                    return name
            else: 
                errors_str += 'Request error code: '+str(req.status_code)+'\n'
        except Exception as e:
            #print(e)
            errors_str += str(e)+'\n'
        t1 = time.time()
        if (t1-t0) < 1:
            time.sleep(t1-t0)

    if i == n_retry:
        flog.write(errors_str)
        
    return name

if __name__ == '__main__':
    fdir_data = '../../data/databases/'
    fname = fdir_data+"log_download_PubChem_synonyms.out"
    flog = open(fname, 'w')
    all_substrates = np.load(fdir_data+'all_substrate_names.npy')
    start = time.time()
    substrate_map = {}

    for substrate in tqdm.tqdm(all_substrates):
        t0 = time.time()
        substrate_name = get_substrate_synonym(substrate, flog)
        substrate_map[substrate] = substrate_name
        t1 = time.time()
        if (t1-t0) < 1:
            time.sleep(t1-t0)

    end = time.time()
    print("Time: ", end - start)
    flog.close()

    with open(fdir_data+'substrate_synonym_map_NEW.csv','w') as f:
        w = csv.writer(f)
        w.writerows(substrate_map.items())