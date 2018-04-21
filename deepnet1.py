#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:33:05 2018

@author: megoconnell
"""
#https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/deepnet.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import merge
import keras.layers.merge
from keras.layers import *
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

data1 = pd.read_csv('/Users/megoconnell/Documents/Courses/Independent Study/RNNs/Item_Item_Same_Paper.csv', sep='\t')


tk = text.Tokenizer(num_words=200000)

data = data1.loc[data1['fold_id'].isin([0,1,2,3])] 
test_data = data1.loc[data1['fold_id'].isin([4])]
y = data.is_duplicate.values
yt = test_data.is_duplicate.values

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = np.asarray(sequence.pad_sequences(x1, maxlen=max_len))

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = np.asarray(sequence.pad_sequences(x2, maxlen=max_len))

tk.fit_on_texts(list(test_data.question1.values) + list(test_data.question2.values.astype(str)))
t1 = tk.texts_to_sequences(test_data.question1.values)
t1 = np.asarray(sequence.pad_sequences(t1, maxlen=max_len))

t2 = tk.texts_to_sequences(test_data.question2.values.astype(str))
t2 = np.asarray(sequence.pad_sequences(t2, maxlen=max_len))

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)
ytest_enc = np_utils.to_categorical(yt)
#embeddings_index = {}
#f = open('/Users/megoconnell/Documents/Courses/Independent Study/RNNs/glove.840B.300d.txt')
#for line in tqdm(f):
    #values = line.split(',')
    #word = values[0]
    #print(word)
    #coefs = np.asarray(values[1:])
    #print(coefs)
    #embeddings_index[word] = coefs
    #if len(embeddings_index.get(word))>300:
        #print('here is your problem',i)
#f.close()
#embeddings_index.get('word')
#print('Found %s word vectors.' % len(embeddings_index))

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/megoconnell/Documents/Courses/Independent Study/RNNs/GoogleNews-vectors-negative300.bin', binary=True)
words = model.wv

import re
all_words = [word for sentence in data1['question1'] for word in re.split('\W', sentence)]
unique = (set(all_words))

a = unique
c = []
for word in a:
    if word in words.vocab:
        c.append(word)

embeddings_index = {}
for word in c:
    vword = model[word]
    vector = {word : vword }
    embeddings_index.update(vector)



embedding_matrix = np.zeros((len(word_index)+1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#embeddings_index.get('the')        

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)


print("Starting training at", datetime.datetime.now())
tt0 = time.time()
merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=384, nb_epoch=25,
                 verbose=1, validation_data=([t1, t2, t1, t2, t1, t2], yt),
                 shuffle=True, callbacks=[checkpoint])
tt1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((tt1 - tt0) / 60.))