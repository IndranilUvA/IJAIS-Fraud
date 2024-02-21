import pandas as pd

raw_data= pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data/Compustat data/Compustat raw.csv")
rusboost_vars = ["csho","act","sstk","ppegt","ap","che","prcc_f",
                 "re","invt","ceq","dlc","dp","rect","cogs",
                 "at","dltis","ib","dltt","xint","txt","lct",
                 "sale","txp","ivao","lt","ivst","ni","pstk"]

id_vars = ["fyear","cik"]

req_vars = rusboost_vars + id_vars
data_small = raw_data[req_vars]
text_data = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/Full Set/final_text_data.csv")

data_small = data_small.dropna(how='any', subset=['cik'])
data_small.cik = data_small.cik.astype(int)
new_df = pd.merge(text_data, data_small,  how='left', left_on=['year','cik'], right_on = ['fyear','cik'])
compustat_vars = rusboost_vars + id_vars + ["fraud"]
new_df=new_df[compustat_vars]
comp = new_df.drop_duplicates(['cik','fyear'],keep= 'last')
print(comp.groupby(["fyear"])["fraud"].value_counts())
comp.to_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data/Compustat data/Compustat_fraud.csv", index = False)