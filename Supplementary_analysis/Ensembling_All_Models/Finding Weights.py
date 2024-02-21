import pandas as pd
from sklearn.metrics import roc_auc_score

i = 1999
bert_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/Bert/"
lda_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/LDA/"
rusboost_folder = "C:/Users/indra/Desktop/NLP Ana paper3/CIKM 2021/Results/RUSboost/"

bert_df = pd.read_csv(bert_folder + "test_" + str(i) + ".csv")
rusboost_df = pd.read_csv(rusboost_folder + "RUSboost_test_" + str(i) + ".csv")
lda_df = pd.read_csv(lda_folder + "LDA_test_" + str(i) + ".csv")

auc = 0
weight = 0
for j in [j/10 for j in range(1,10)]:
    new_df = pd.merge(bert_df, rusboost_df,  how='right', left_on=['cik'], right_on = ['cik'])
    cols = ["cik","pred","pred_rusboost","fraud_y"]
    new_df['BERT_rank'] = new_df['pred'].rank()
    new_df['RUSboost_rank'] = new_df['pred_rusboost'].rank()
    new_df["average_rank"] = new_df['BERT_rank']*j + new_df['RUSboost_rank']*(1-j)
    if roc_auc_score(new_df["fraud_y"], new_df["average_rank"]) > auc:
        auc = roc_auc_score(new_df["fraud_y"], new_df["average_rank"])
        weight = j
print("Optimal weight for BERT is", weight," and optimal weight for RUSBoost model is", round(1-weight,1))

auc = 0
weight1 = 0
weight2 = 0
for i in [i/10 for i in range(1,10)]:
    for j in [j/10 for j in range(1,10)]:
        if i+j<1:
            k = round(1-i-j,1)
            new_df_1 = pd.merge(bert_df, rusboost_df,  how='right', left_on=['cik'], right_on = ['cik'])
            new_df = pd.merge(lda_df, new_df_1,  how='right', left_on=['cik'], right_on = ['cik'])
            cols = ["cik","pred","pred_rusboost","pred_lda","fraud_y"]
            new_df['BERT_rank'] = new_df['pred'].rank()
            new_df['RUSboost_rank'] = new_df['pred_rusboost'].rank()
            new_df['LDA_rank'] = new_df['pred_lda'].rank()
            new_df["average_rank"] = new_df['BERT_rank']*i + new_df['RUSboost_rank']*j + new_df['LDA_rank']*k
            if roc_auc_score(new_df["fraud_y"], new_df["average_rank"]) > auc:
                auc = roc_auc_score(new_df["fraud_y"], new_df["average_rank"])
                weight1 = i
                weight2 = j
print("Optimal weight for BERT is", weight1, "optimal weight for RUSBoost is", weight2, "and optimal weight for LDA is", round(1-weight1-weight2,1))