import requests
import tqdm
import time
import numpy as np

def download_sabio():
    # Get the list of all EntryIDs
    ENTRYID_QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs'
    query_string = "EntryID:*"
    query = {'format':'txt', 'q':query_string}
    request = requests.get(ENTRYID_QUERY_URL, params = query)
    request.raise_for_status() # raise if 404 error
    entryIDs = [int(x) for x in request.text.strip().split('\n')]
    print('Number of database entries:', len(entryIDs))

    # Make requests in small chunks depending on EntryIDs
    # Otherwise running with "EntryID:*" can result in 404 error
    chunk_size = 400
    _entryIDs = np.split(entryIDs, np.arange(chunk_size, len(entryIDs), chunk_size))
    qs = ["[" + str(i[0]) + " TO " + str(i[-1]) + "]" for i in _entryIDs]

    QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv'
    query = {'fields[]':['EntryID', 'PubMedID', 'Organism', 'Substrate', 'EnzymeType', 'Enzymename', 'UniprotID', 'ECNumber', 'Parameter', 'pH', 'Temperature']}

    fname = "data/databases/Sabio-RK/dataset_download.tsv"
    fout = open(fname, 'w')
    fname = "data/databases/Sabio-RK/temp/log_download.out"
    flog = open(fname, 'w')

    n_retry = 10
    log_tries = [n_retry] * len(qs)
    i = 0
    start = time.time()

    # Connectivity issues (error 500) can break things. If rerunning, check that no artefacts (extra columns) were added
    for q in tqdm.tqdm(qs):
        i += 1
        query_string = "EntryID:"+q
        query['q'] = query_string
        flog.write('-------------\nEntryID chunk index: '+str(i)+'\n')
        for j in range(1, n_retry+1):
            flog.write('Try '+str(j)+'\n') 
            try:
                request = requests.get(QUERY_URL, params = query)
                if request.status_code == 200:
                    ind = 0 if i==1 else 1
                    fout.write(''.join(request.text.splitlines(keepends=True)[ind:]))
                    log_tries[i-1] = j
                    break
                else:
                    flog.write('Request error code: '+str(request.status_code)+'\n')
            except Exception as e:
                flog.write(str(e)+'\n')
   
    end = time.time()
    flog.write('Runtime: '+str(end-start)) 
    fout.close()
    flog.close()

    fname = 'data/databases/Sabio-RK/temp/log_tries.out'
    with open(fname, 'w') as flogntries:
        flogntries.write("\n".join(str(ntry) for ntry in log_tries))


if __name__ == '__main__' :
    download_sabio()