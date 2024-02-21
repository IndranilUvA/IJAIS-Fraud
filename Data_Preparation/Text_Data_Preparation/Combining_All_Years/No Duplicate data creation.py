import pandas as pd

base_folder = "C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/"
for i in range(1994,2014):
    target_file = base_folder + str(i) + "/final_" + str(i) +"_text.csv"
    file = pd.read_csv(target_file)
    print("*"*50, "year", i,"*"*50)
    print(file.shape)
    print("number of unique CIKs", file.cik.nunique())
    print("number of frauds before removing duplicates", file.fraud.sum())
    
    print("now removing duplicate CIKs")
    
    df = file.drop_duplicates(subset='cik', keep="last")
    print(df.shape)
    print("number of unique CIKs", df.cik.nunique())
    print("number of frauds after removing duplicates", df.fraud.sum())
    
    df.to_csv(base_folder + str(i) + "/final_" + str(i) +"_text_nodup.csv", index = False)