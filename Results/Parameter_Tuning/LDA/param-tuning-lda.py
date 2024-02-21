import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
import os
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from scipy import sparse
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from sklearn.linear_model import LogisticRegression
import random
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

t1 = time.time()
data = pd.read_csv("../input/accounting-fraud-detection/final_text_data.csv")
print(data.shape)
data.head()
t2=time.time()
print(t2-t1)

i = 1999

train = data[(data["year"]>i-6) & (data["year"]<i)]
test = data[data["year"] == i]
print(train.shape, test.shape)

print(train.year.value_counts())
print(train.fraud.sum(), test.fraud.sum())

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

cols = ["text","fraud"]
train = train[cols]
test = test[cols]

train['text'] = train['text'].apply(lambda x: remove_stopwords(x))
test['text'] = test['text'].apply(lambda x: remove_stopwords(x))

train['text'] = train.text.apply(lemmatize_text)
test['text'] = test.text.apply(lemmatize_text)

train['text'] = train['text'].apply(lambda x: " ".join(x))
test['text'] = test['text'].apply(lambda x: " ".join(x))

train['text'] = train['text'].str.replace('.','',regex=True)
test['text'] = test['text'].str.replace('.','',regex=True)

train_docs = train["text"].tolist()
test_docs = test["text"].tolist()

cv = CountVectorizer(min_df = 0.1,
                     max_df = 0.9)
sparse_train = cv.fit_transform(train_docs)
sparse_test  = cv.transform(test_docs)

full_sparse_data =  sparse.vstack([sparse_train, sparse_test])
y_train = train.fraud.values

corpus_data_gensim = gensim.matutils.Sparse2Corpus(full_sparse_data, documents_columns=False)
vocabulary_gensim = {}
for key, val in cv.vocabulary_.items():
    vocabulary_gensim[val] = key
    
dict = Dictionary()
dict.merge_with(vocabulary_gensim)

number = []
AUC = []

for i in range(10,150):
    lda = LdaModel(corpus_data_gensim, num_topics = i, random_state = seed)

    def document_to_lda_features(lda_model, document):
        topic_importances = lda.get_document_topics(document, minimum_probability=0)
        topic_importances = np.array(topic_importances)
        return topic_importances[:,1]

    lda_features = list(map(lambda doc:document_to_lda_features(lda, doc),corpus_data_gensim))

    data_pd_lda_features = pd.DataFrame(lda_features)
    data_pd_lda_features.columns = ["topic"+str(j) for j in range(i)]
    data_pd_lda_features_train = data_pd_lda_features.iloc[:y_train.shape[0]]
    data_pd_lda_features_test = data_pd_lda_features.iloc[y_train.shape[0]:]
    logmodel = LogisticRegression()
    logmodel.fit(data_pd_lda_features_train,y_train)
    predictions = logmodel.predict_proba(data_pd_lda_features_test)[:,1]

    print("AUC for number of topics",i,"is",roc_auc_score(test["fraud"], predictions))
    number.append(i)
    AUC.append(roc_auc_score(test["fraud"], predictions))

d = {'number_of_topics':number,'AUC':AUC}
df = pd.DataFrame(d)
print(df)

print("optimum number of topics is", df[df.AUC == max(AUC)]["number_of_topics"].tolist())
df.to_csv("LDA parameter tuning.csv", index = False)