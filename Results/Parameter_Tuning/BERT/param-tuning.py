debug = False
import requests
url = "https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py"

response = requests.get(url)

if response.status_code == 200:
    with open("tokenization.py", "wb") as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print("Failed to download file. Status code:", response.status_code)

import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import random
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub


seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

!pip install bert-for-tf2
from bert import bert_tokenization

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

t1 = time.time()
data = pd.read_csv("../input/accounting-fraud-detection/final_text_data.csv")
print(data.shape)
data.head()
t2=time.time()
print("reading the entire data takes",t2-t1,"seconds")

i = 1999

train = data[(data["year"]>i-6) & (data["year"]<i)]
test = data[data["year"] == i]
print("-"*50,"test year",i,"-"*50)
print(train.shape, test.shape)
print(train.year.value_counts().sort_index())
print(test.year.value_counts())
print(train.fraud.sum(), test.fraud.sum())

if debug == True:
    train = train.tail(100)
    test = test.head(100)

print(train.shape, test.shape, train.fraud.sum(), test.fraud.sum())

def bert_encode(texts, tokenizer, max_len=512, first=True):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        if first == True:
            text = text[:max_len-2]
        else: 
            text = text[-(max_len-2):]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

train_labels = train.fraud.values
test_labels = test.fraud.values

train_input_first = bert_encode(train.text.values,tokenizer, first=True)
test_input_first = bert_encode(test.text.values, tokenizer, first=True)

train_input_last = bert_encode(train.text.values,tokenizer, first=False)
test_input_last = bert_encode(test.text.values, tokenizer, first=False)

model = build_model(bert_layer, max_len=512)
print(model.summary())

batch_size = []
epochs = []
AUC_first = []
AUC_last = []
AUC_average = []
for bs in [2,4,8]:
    for ep in [2,4,8]:
        print("batch size =", bs, "and number of epochs =", ep)
        checkpoint_first = ModelCheckpoint('model_first.h5', monitor='val_loss', save_best_only=False)
        train_history_first = model.fit(train_input_first, train_labels,validation_data=(test_input_first, test_labels),epochs=ep,callbacks=[checkpoint_first],batch_size=bs)
        test_pred_first = model.predict(test_input_first)

        print("First 512 tokens: AUC value for year", i, "is",roc_auc_score(test["fraud"], test_pred_first))

        checkpoint_last = ModelCheckpoint('model_last.h5', monitor='val_loss', save_best_only=False)
        train_history_last = model.fit(train_input_last, train_labels,validation_data=(test_input_last, test_labels),epochs=ep,callbacks=[checkpoint_last],batch_size=bs)
        test_pred_last = model.predict(test_input_last)

        print("Last 512 tokens: AUC value for year", i, "is",roc_auc_score(test["fraud"], test_pred_last))

        print("Rank average AUC for year ", i , roc_auc_score(test["fraud"], rankdata(test_pred_first)*0.5 + rankdata(test_pred_last)*0.5))
        print("*"*100)
        batch_size.append(bs)
        epochs.append(ep)
        AUC_first.append(roc_auc_score(test["fraud"], test_pred_first))
        AUC_last.append(roc_auc_score(test["fraud"], test_pred_last))
        AUC_average.append(roc_auc_score(test["fraud"], rankdata(test_pred_first)*0.5 + rankdata(test_pred_last)*0.5))

dict = {'batch_size': batch_size, 'epochs': epochs, 'AUC_first': AUC_first, 'AUC_last': AUC_last, 'AUC_average': AUC_average}
param_df = pd.DataFrame(dict)
print(param_df)