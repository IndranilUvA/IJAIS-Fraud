import numpy as np
import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import roc_auc_score
import random
random.seed(0)
np.random.seed(0)

data = pd.read_csv("C:/Users/indra/Desktop/NLP Ana paper3/Data/Compustat data/Compustat_fraud.csv")

i = 1999

train = data[(data["fyear"]>i-6) & (data["fyear"]<i)]
test = data[data["fyear"] == i]

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

rusboost = RUSBoostClassifier(n_estimators=3000, ## 3000
                              learning_rate=0.1,
                              algorithm='SAMME.R',
                              sampling_strategy = train.fraud.sum()/(train.shape[0]-train.fraud.sum()),
                              random_state=0)

comp_cols = ['csho', 'act', 'sstk', 'ppegt', 'ap', 'che', 'prcc_f', 're', 'invt',
            'ceq', 'dlc', 'dp', 'rect', 'cogs', 'at', 'dltis', 'ib', 'dltt', 'xint',
            'txt', 'lct', 'sale', 'txp', 'ivao', 'lt', 'ivst', 'ni', 'pstk']
X_train = train[comp_cols]
y_train = train.fraud.values
rusboost.fit(X_train, y_train)

X_test = test[comp_cols]
y_pred = rusboost.predict_proba(X_test)[:,1]

print("AUC for year", i,"is ", roc_auc_score(test["fraud"], y_pred))
number = []
AUC = []
for j in range(100,10000,100):
    rusboost = RUSBoostClassifier(n_estimators=j,
                                  learning_rate=0.1,
                                  algorithm='SAMME.R',
                                  sampling_strategy = train.fraud.sum()/(train.shape[0]-train.fraud.sum()),
                                  random_state=0)

    comp_cols = ['csho', 'act', 'sstk', 'ppegt', 'ap', 'che', 'prcc_f', 're', 'invt',
                'ceq', 'dlc', 'dp', 'rect', 'cogs', 'at', 'dltis', 'ib', 'dltt', 'xint',
                'txt', 'lct', 'sale', 'txp', 'ivao', 'lt', 'ivst', 'ni', 'pstk']
    X_train = train[comp_cols]
    y_train = train.fraud.values
    rusboost.fit(X_train, y_train)

    X_test = test[comp_cols]
    y_pred = rusboost.predict_proba(X_test)[:,1]

    print("validation AUC for number of trees", j ,"is ", roc_auc_score(test["fraud"], y_pred))
    number.append(j)
    AUC.append(roc_auc_score(test["fraud"], y_pred))

d = {'number_of_trees':number,'AUC':AUC}
df = pd.DataFrame(d)
print(df)
print("optimum number of trees is", df[df.AUC == max(AUC)]["number_of_trees"].tolist())

df.to_csv("C:/Users/indra/Desktop/NLP Ana paper3/Results/Output/RUSBoost parameter tuning.csv", index = False)