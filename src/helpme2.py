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



def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    # text = re.sub(r"tom", "", text)
    # text = re.sub(r"톰은", "", text)
    # text = re.sub(r"톰이", "", text)
    # text = re.sub(r"톰을", "", text)
    # text = re.sub(r"톰한테", "", text)
    # text = re.sub(r"톰의", "", text)
    # text = re.sub(r"톰과", "", text)
    # text = re.sub(r"톰", "", text)
    # text = re.sub(r"톰도", "", text)



    text = re.sub(r"’", "'", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"([?.!,¿])", r"", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
#     text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    return text


def start_end_tagger(decoder_input_sentence):
    start_tag = "<start> "
    end_tag = " <end>"
    final_target = start_tag + decoder_input_sentence + end_tag
    return final_target


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    padded_tensor = pad_sequences(tensor, maxlen=max_length(tensor), padding='post',truncating='pre')

    return padded_tensor, lang_tokenizer


def preprocess(language):
    language = language.apply(clean_text)
    
    language = language.apply(start_end_tagger)
    
    return language

def preprocess_sentence(sentence):
    sentence = clean_text(sentence)
    sentence = start_end_tagger(sentence)
    return sentence

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_sequence(model, tokenizer, source):
    source = source.reshape((1, source.shape[0]))
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word == '<end>':
            break
        target.append(word)
    return ' '.join(target)