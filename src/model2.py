import pandas as pd
import numpy as np
# from langdetect import detect
import re
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, GRU, Embedding, Layer
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
# import pydot

from helpme import clean_text,start_end_tagger,max_length,tokenize,preprocess,preprocess_sentence



df_all = pd.read_csv('data/final_df_fix.txt',sep='\t')

eng = preprocess(df_all['eng'])
kor = preprocess(df_all['kor'])

input_tensor, input_lang_tokenizer = tokenize(eng)
target_tensor, target_lang_tokenizer = tokenize(kor)

eng_vocab_size = len(input_lang_tokenizer.word_index)+1
kor_vocab_size = len(target_lang_tokenizer.word_index)+1

eng_max_length = len(input_tensor[0])
kor_max_length = len(target_tensor[0])

encoder_input = Input(shape=(None,),name='Encoder_input')
embedding_dim=50
embedded_input = Embedding(input_dim=eng_vocab_size,
                           output_dim=embedding_dim,
                           name='Embedding_layer')(encoder_input)
encoder_lstm = LSTM(units=50,
                   activation='relu',
                   return_sequences=False,
                   return_state=True,
                   name='Encoder_lstm')
encoder_out, enc_h_state, enc_c_state = encoder_lstm(embedded_input)

decoder_input = Input(shape=(None,1), name='Decoder_input')
# embedded_decoder = Embedding(kor_vocab_size,
#                             100,
#                             name='Decoder_embedded_layer')(decoder_input)
decoder_lstm = LSTM(units=50,
                   activation='relu',
                   return_sequences=True,
                   return_state=True,
                   name='Decoder_lstm')
decoder_out,_,_ = decoder_lstm(decoder_input,initial_state=[enc_h_state,enc_c_state])

final_dense = Dense(kor_vocab_size,activation='softmax',name='Final_dense_layer')
logits = final_dense(decoder_out)

model = Model([encoder_input,decoder_input],logits)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])


decoder_kor_input = target_tensor.reshape((-1,kor_max_length,1))[:,:-1,:]
decoder_kor_target = target_tensor.reshape((-1,kor_max_length,1))[:,1:,:]

model.fit([input_tensor,decoder_kor_input],decoder_kor_target,
         epochs=15,
         batch_size=20,
         validation_split=0.2)

inf_encoder_model = Model(encoder_input, [enc_h_state, enc_c_state])

decoder_initial_states = [Input(shape=(50,)),
                         Input(shape=(50,))]

decoder_output, dec_h_state, dec_c_state = decoder_lstm(decoder_input, initial_state=decoder_initial_states)

logits = final_dense(decoder_output)

inf_decoder_model = Model([decoder_input] + decoder_initial_states, [logits,dec_h_state, dec_c_state])

kor_id2word = {idx:word for word, idx in target_lang_tokenizer.word_index.items()}

def translate(sentence):
    sentence = preprocess_sentence(sentence)
    
    input_sentence = [input_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    input_sentence = pad_sequences([input_sentence],maxlen=eng_max_length,padding='post')
    input_sentence_tensor = tf.convert_to_tensor(input_sentence)
    return input_sentence_tensor