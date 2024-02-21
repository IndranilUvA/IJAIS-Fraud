import pandas as pd

base_folder = "C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/"

data_list = []
for i in range(1994,2014):
    target_file = base_folder + str(i) + "/final_" + str(i) +"_text_nodup.csv"
    file = pd.read_csv(target_file)
    file["outlier"] = 0
    data_list.append(file)

outlier_data = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/Full Set/manual_fraud_text.csv")
outlier_data["fraud"]=1
outlier_data["outlier"]=1
cols = ['cik', 'filing_date', 'url', 'month', 'year', 'date', 'text', 'fraud','outlier']
outlier_data = outlier_data[cols]
data_list.append(outlier_data)

new_df = pd.concat(data_list)
print(new_df.shape)
print("total number of fraud samples", new_df.fraud.sum())
print("number of fraud samples to be removed", new_df["text"].isnull().sum())
new_df = new_df[new_df['text'].notna()]
print("Full data size", new_df.shape)

new_df.groupby(["year"])["fraud"].value_counts()
new_df.to_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data Preparation/Full Set/final_text_data.csv", index = False)