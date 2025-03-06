import multiprocessing as mp
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from functools import partial
import time
import csv
import re
import tqdm


def init_pool():
    global session
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    print(f"init - {mp.current_process()}  session id: {id(session)}")


def get_UniProtID(query, requrl):
    req = session.get(requrl+query)
    #print(f"worker - {mp.current_process()}  session id: {id(session)}")
    if req.status_code != 200:
        print(req.status_code)
        return ''
    else:
        uniprotid = ''
        for line in req.text.split('\n'):
            if line.startswith('>'):
                if uniprotid == '':
                    uniprotid = re.search("\\|.*\\|", line).group()[1:-1]
                else:
                    return ''
        return uniprotid
    

if __name__ == '__main__':
    fdir = '../../data/databases/'
    requrl = "https://rest.uniprot.org/uniprotkb/search?format=fasta&query="
    all_queries = np.load(fdir+'all_ECNumber_organism_queries.npy')#[:100]
    start = time.time()
    query_map = {}

    mp.set_start_method('spawn')
    pool = mp.Pool(10, initializer=init_pool )
    res = tqdm.tqdm(pool.imap(partial(get_UniProtID, requrl=requrl), all_queries), total=len(all_queries))
    
    for i, uniprotid in enumerate(res):
        query = all_queries[i]
        query_map[query] = uniprotid

    end = time.time()
    print("Time: ", end - start)

    #print(query_map)
    with open(fdir+'UniProtKB_query_map_FINAL.csv','w') as f:
        w = csv.writer(f)
        w.writerows(query_map.items())