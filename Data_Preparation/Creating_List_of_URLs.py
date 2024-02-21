import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

header_list = ["ticker", "cik"]
all_cik = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/ticker-cik.txt", sep='\t',
                      names=header_list)

all_cik["cik"] = all_cik.cik.astype(str)
cik_list = list(set(all_cik.cik.tolist()))

def cik_to_all_urls(cik):
    base = "https://www.sec.gov/Archives/edgar/data/"
    endpoint = r"https://www.sec.gov/cgi-bin/browse-edgar"

    param_dict = {'action': 'getcompany',
                  'CIK': cik,
                  'type': '10-k',
                  'dateb': '20140101',
                  'owner': 'exclude',
                  'start': '',
                  'output': 'atom',
                  'count': '100'}

    response = requests.get(url=endpoint, params=param_dict)
    soup = BeautifulSoup(response.content, 'lxml')
    df_cik = pd.DataFrame({'cik': [], 'filing_type': [], 'accession_num': [], 'filing_date': [], 'url': []})

    entries = soup.find_all("entry")

    for entry in entries:
        accession_num = entry.find("accession-number").text
        filing_type = entry.find("filing-type").text
        filing_date = entry.find("filing-date").text
        url = base + cik + "/" + accession_num + ".txt"
        df_cik = df_cik.append({'cik': cik,
                                'filing_type': filing_type,
                                'accession_num': accession_num,
                                'filing_date': filing_date,
                                'url': url}, ignore_index=True)

    return df_cik

start = time.time()
col_names = ['cik', 'filing_type', 'accession_num', 'filing_date', 'url']
master_df = pd.DataFrame(columns=col_names)

for c in tqdm(cik_list):
    master_df = master_df.append(cik_to_all_urls(c))
end = time.time()

print("time taken to create the master dataset is", end - start, " seconds")
master_df.to_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/master_df.csv", index=False)