import pandas as pd
import numpy as np
from langdetect import detect
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
from tensorflow.keras.utils import plot_model
import pydot


class Encoder(Model):
    def __init__(self, vocab_size, embed_dim, enc_units, batch_size):
        super(Encoder,self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.enc_units = enc_units
        self.batch_size = batch_size
        
        self.embedding_layer = Embedding(self.vocab_size,self.embed_dim)
        self.gru_layer = GRU(self.enc_units, return_sequences=True,return_state=True)
        
    def call(self,x,hidden):
        x = self.embedding_layer(x)
        output,state = self.gru_layer(x,initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size,self.enc_units))


class Attention(Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        
    def call(self, query, values):
        hidden_with_time = tf.expand_dims(query,1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time)))
        
        attention_weights = tf.nn.softmax(score,axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector,axis=1)
        
        return context_vector, attention_weights


class Decoder(Model):
    def __init__(self, vocab_size, embed_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dec_units = dec_units
        self.batch_size = batch_size
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.gru_layer = GRU(self.dec_units, return_sequences=True,return_state=True)
        self.dense = Dense(vocab_size)
        
        self.attention = Attention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        attention_vector, attention_weights = self.attention(hidden, enc_output)
        
        x = self.embedding_layer(x)
        
        x = tf.concat([tf.expand_dims(attention_vector, 1),x],axis=-1)
        
        output, state = self.gru_layer(x)
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.dense(output)
        
        return x, state, attention_weights