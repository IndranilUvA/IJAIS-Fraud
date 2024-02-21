import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

bert_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/Bert/"
lda_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/LDA/"
rusboost_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/RUSboost/"
bert_del_words_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/Bert_Del_Words/"

BERT_AUC_first = []
BERT_AUC_last = []
BERT_AUC = []
BERT_capture = []
print("--"*20,"printing BERT AUCs","--"*20)
for i in range(2000,2014):
    df = pd.read_csv(bert_folder + "test_" + str(i) + ".csv")
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred_first"]),"--"*10)
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred_last"]),"--"*10)
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred"]),"--"*10)
    print("--"*10,"number of fraud cases captured in top 1% data", df[df['pred']>np.percentile(df['pred'], 99)].fraud.sum(), "out of total fraud samples", df["fraud"].sum(),"--"*10)
    BERT_AUC_first.append(roc_auc_score(df["fraud"], df["pred_first"]))
    BERT_AUC_last.append(roc_auc_score(df["fraud"], df["pred_last"]))
    BERT_AUC.append(roc_auc_score(df["fraud"], df["pred"]))
    BERT_capture.append(df[df['pred']>np.percentile(df['pred'], 99)].fraud.sum())

print("Average AUC using only 512 tokens",sum(BERT_AUC_first)/len(BERT_AUC_first))
print("Average AUC using last 512 tokens",sum(BERT_AUC_last)/len(BERT_AUC_last))
print("Rank average AUC",sum(BERT_AUC)/len(BERT_AUC))
print("Total number of fraud samples captured in top 1% data is", sum(BERT_capture))

LDA_AUC = []
LDA_capture = []
print("--"*20,"printing LDA AUCs","--"*20)
for i in range(2000,2014):
    df = pd.read_csv(lda_folder + "LDA_test_" + str(i) + ".csv")
    print("--"*10,"LDA AUC for year",i,roc_auc_score(df["fraud"], df["pred_lda"]),"--"*10)
    print("--"*10,"number of fraud cases captured in top 1% data", df[df['pred_lda']>np.percentile(df['pred_lda'], 99)].fraud.sum(), "out of total fraud samples", df["fraud"].sum(),"--"*10)
    LDA_AUC.append(roc_auc_score(df["fraud"], df["pred_lda"]))
    LDA_capture.append(df[df['pred_lda']>np.percentile(df['pred_lda'], 99)].fraud.sum())

print("Average LDA AUC",sum(LDA_AUC)/len(LDA_AUC))
print("Total number of fraud samples captured in top 1% data is", sum(LDA_capture))

RUSBoost_AUC = []
RUSBoost_capture = []
print("--"*20,"printing RUSBoost AUCs","--"*20)
for i in range(2000,2014):
    df = pd.read_csv(rusboost_folder + "RUSboost_test_" + str(i) + ".csv")
    print("--"*10,"LDA AUC for year",i,roc_auc_score(df["fraud"], df["pred_rusboost"]),"--"*10)
    print("--"*10,"number of fraud cases captured in top 1% data", df[df['pred_rusboost']>np.percentile(df['pred_rusboost'], 99)].fraud.sum(), "out of total fraud samples", df["fraud"].sum(),"--"*10)
    RUSBoost_AUC.append(roc_auc_score(df["fraud"], df["pred_rusboost"]))
    RUSBoost_capture.append(df[df['pred_rusboost']>np.percentile(df['pred_rusboost'], 99)].fraud.sum())

print("Average RUSBoost AUC",sum(RUSBoost_AUC)/len(RUSBoost_AUC))
print("Total number of fraud samples captured in top 1% data is", sum(RUSBoost_capture))

BERT_AUC_subset = []
BERT_subset_capture = []
for i in range(2000,2014):
    bert_df = pd.read_csv(bert_folder + "test_" + str(i) + ".csv")
    rusboost_df = pd.read_csv(rusboost_folder + "RUSboost_test_" + str(i) + ".csv")
    new_df = pd.merge(bert_df, rusboost_df,  how='right', left_on=['cik'], right_on = ['cik'])
    cols = ["cik","pred","pred_rusboost","fraud_y"]
    print("year",i,"RUSBOOST AUC",roc_auc_score(new_df["fraud_y"], new_df["pred_rusboost"]),"BERT AUC",roc_auc_score(new_df["fraud_y"], new_df["pred"]))
    print("-"*10,"number of fraud cases captured in top 1% subset ensemble data", new_df[new_df['pred']>np.percentile(new_df['pred'], 99)].fraud_y.sum(), "out of total fraud samples", new_df["fraud_y"].sum(),"-"*10)
    BERT_AUC_subset.append(roc_auc_score(new_df["fraud_y"], new_df["pred"]))
    BERT_subset_capture.append(new_df[new_df['pred']>np.percentile(new_df['pred'], 99)].fraud_y.sum())

print("Average BERT AUC on ensemble data",sum(BERT_AUC_subset)/len(BERT_AUC_subset))
print("Total number of fraud samples captured in top 1% data by BERT predictions on ensemble data is", sum(BERT_subset_capture))

Average_AUC_subset = []
Average_subset_capture = []
for i in range(2000,2014):
    bert_df = pd.read_csv(bert_folder + "test_" + str(i) + ".csv")
    rusboost_df = pd.read_csv(rusboost_folder + "RUSboost_test_" + str(i) + ".csv")
    new_df = pd.merge(bert_df, rusboost_df,  how='right', left_on=['cik'], right_on = ['cik'])
    cols = ["cik","pred","pred_rusboost","fraud_y"]
    new_df['BERT_rank'] = new_df['pred'].rank()
    new_df['RUSboost_rank'] = new_df['pred_rusboost'].rank()
    new_df["average_rank"] = new_df['BERT_rank']*0.5 + new_df['RUSboost_rank']*0.5
    print("--"*10,"Ensemble AUC for year",i,roc_auc_score(new_df["fraud_y"], new_df["average_rank"]),"--"*10)
    print("-"*5,"number of fraud cases captured in top 1% subset ensemble data by ensemble model", new_df[new_df['average_rank']>np.percentile(new_df['average_rank'], 99)].fraud_y.sum(), "out of total fraud samples", new_df["fraud_y"].sum(),"-"*5)
    Average_AUC_subset.append(roc_auc_score(new_df["fraud_y"], new_df["average_rank"]))
    Average_subset_capture.append(new_df[new_df['average_rank']>np.percentile(new_df['average_rank'], 99)].fraud_y.sum())


print("Average RUSBoost + BERT AUC",sum(Average_AUC_subset)/len(Average_AUC_subset))
print("Total number of fraud samples captured in top 1% data by RUSBoost + BERT is", sum(Average_subset_capture))

BERT_Replaced_AUC_first = []
BERT_Replaced_AUC_last = []
BERT_Replaced_AUC = []
BERT_Replaced_capture = []
print("--"*20,"printing BERT AUCs","--"*20)
for i in range(2000,2014):
    df = pd.read_csv(bert_del_words_folder + "test_" + str(i) + "_del_words.csv")
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred_first"]),"--"*10)
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred_last"]),"--"*10)
    print("--"*10,"BERT AUC for year",i,roc_auc_score(df["fraud"], df["pred_last"]),"--"*10)
    print("--"*10,"number of fraud cases captured in top 1% data", df[df['pred']>np.percentile(df['pred'], 99)].fraud.sum(), "out of total fraud samples", df["fraud"].sum(),"--"*10)
    BERT_Replaced_AUC_first.append(roc_auc_score(df["fraud"], df["pred_first"]))
    BERT_Replaced_AUC_last.append(roc_auc_score(df["fraud"], df["pred_last"]))
    BERT_Replaced_AUC.append(roc_auc_score(df["fraud"], df["pred"]))
    BERT_Replaced_capture.append(df[df['pred']>np.percentile(df['pred'], 99)].fraud.sum())

print("Average AUC for replaced words using first 512 tokens",sum(BERT_Replaced_AUC_first)/len(BERT_Replaced_AUC_first))
print("Average AUC for replaced words using last 512 tokens",sum(BERT_Replaced_AUC_last)/len(BERT_Replaced_AUC_last))
print("Rank average AUC for replaced words",sum(BERT_Replaced_AUC)/len(BERT_Replaced_AUC))
print("Total number of fraud samples captured in top 1% data for replaced words is", sum(BERT_Replaced_capture))

Average_all_AUC_subset = []
Average_all_subset_capture = []
for i in range(2000,2014):
    bert_df = pd.read_csv(bert_folder + "test_" + str(i) + ".csv")
    rusboost_df = pd.read_csv(rusboost_folder + "RUSboost_test_" + str(i) + ".csv")
    lda_df = pd.read_csv(lda_folder + "LDA_test_" + str(i) + ".csv")
    new_df_1 = pd.merge(bert_df, rusboost_df,  how='right', left_on=['cik'], right_on = ['cik'])
    new_df = pd.merge(lda_df, new_df_1,  how='right', left_on=['cik'], right_on = ['cik'])
    cols = ["cik","pred","pred_rusboost","pred_lda","fraud_y"]
    new_df['BERT_rank'] = new_df['pred'].rank()
    new_df['RUSboost_rank'] = new_df['pred_rusboost'].rank()
    new_df['LDA_rank'] = new_df['pred_lda'].rank()
    new_df["average_rank"] = new_df['BERT_rank']*0.3 + new_df['RUSboost_rank']*0.4 + new_df['LDA_rank']*0.3
    print("--"*10,"Ensemble AUC for year",i,roc_auc_score(new_df["fraud_y"], new_df["average_rank"]),"--"*10)
    print("-"*5,"number of fraud cases captured in top 1% subset ensemble data by ensemble model", new_df[new_df['average_rank']>np.percentile(new_df['average_rank'], 99)].fraud_y.sum(), "out of total fraud samples", new_df["fraud_y"].sum(),"-"*5)
    Average_all_AUC_subset.append(roc_auc_score(new_df["fraud_y"], new_df["average_rank"]))
    Average_all_subset_capture.append(new_df[new_df['average_rank']>np.percentile(new_df['average_rank'], 99)].fraud_y.sum())

print("Rank average AUC for BERT+RUSBoost+LDA",sum(Average_all_AUC_subset)/len(Average_all_AUC_subset))
print("Total number of fraud samples captured in top 1% data for BERT+RUSBoost+LDA", sum(Average_all_subset_capture))