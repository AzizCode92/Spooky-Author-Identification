import numpy as np
import time
import pandas as pd
import csv
from collections import defaultdict

import keras
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(7)

##############  Load train.zip + test.zip ################
train_df = pd.read_csv("../data/train.zip",compression='zip')
test_df = pd.read_csv("../data/test.zip",compression='zip')
author_lst = sorted(list(set(train_df['author'])))
author2idx = dict([(author, idx) for idx, author in enumerate(author_lst)])
y = to_categorical([author2idx[author] for author in train_df['author']])

################## quick process #######################################

def preprocess(text):   # TODO: refine this function (cf. `data_helper.py`)
    text = text.replace("' ", " ' ")  # TODO: remove space in "' "
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )

    return text

def create_docs(data_df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for w_index in range(len(q) - n + 1):
                ngrams.append('--'.join(q[w_index:w_index + n]))
        return q + ngrams

    docs = []
    for doc in data_df['text']:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    return docs


min_count = 2

docs = create_docs(train_df)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([count >= min_count for _, count in tokenizer.word_counts.items()])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)

input_dim = np.max(docs) + 1
embedding_dims = 20



def create_model(embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    # now shape: `(batch_size, sequence_length, output_dim)`
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


epochs = 10
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.20)

timestamp = str(int(time.time()))
model = create_model()
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=2, monitor='val_loss')])
model.save('FastText_model_{}'.format(timestamp))
# model.load_weights('')

test_df = pd.read_csv('../data/test.zip', compression='zip')
docs = create_docs(test_df)
docs = tokenizer.texts_to_sequences(docs)
docs = pad_sequences(sequences=docs, maxlen=maxlen)
test_predict = model.predict_proba(docs)

test_df = test_df.drop(list(set(test_df.columns) - set(['id'])), axis=1)
for author in author_lst:
    test_df[author] = test_predict[:, author2idx[author]]

test_df = test_df[['id'] + author_lst]

test_df.to_csv('predictions_FastText_{0}_1.csv'.format(timestamp), index=False,
quoting=csv.QUOTE_NONNUMERIC)
