import multiprocessing as mp
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
import time
import csv
import tqdm


def init_pool():
    global session
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    print(f"init - {mp.current_process()}  session id: {id(session)}")


def get_UniProtKB_sequence(uniprotid):
    requrl = "https://www.uniprot.org/uniprot/%s.fasta" % uniprotid
    req = session.get(requrl)
    #print(f"worker - {mp.current_process()}  session id: {id(session)}")
    if req.status_code != 200:
        print(req.status_code)
        return None
    else:
        if req.text == '':
            return None
        else:
            seq = "".join(req.text.split("\n")[1:])
            return seq
    

if __name__ == '__main__':
    fdir = '../../data/databases/'
    all_UniProtIDs = np.load(fdir+'all_UniProtIDs.npy')
    start = time.time()
    uniprotid_seq_map = {}

    mp.set_start_method('spawn')
    pool = mp.Pool(8, initializer=init_pool )
    res = tqdm.tqdm(pool.imap(get_UniProtKB_sequence, all_UniProtIDs), total=len(all_UniProtIDs))
    
    for i, seq in enumerate(res):
        uniprotid = all_UniProtIDs[i]
        uniprotid_seq_map[uniprotid] = seq
        
    end = time.time()
    print("Time: ", end - start)

    #print(query_map)
    with open(fdir+'UniProtKB_sequence_map.csv','w') as f:
        w = csv.writer(f)
        w.writerows(uniprotid_seq_map.items())