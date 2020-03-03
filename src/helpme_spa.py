import pandas as pd
import numpy as np
# from langdetect import detect
import re
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, GRU, Embedding, Layer, RepeatVector, TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
# import pydot



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


if __name__ == "__main__":

    print('starting...')

    df = pd.read_csv('spa.txt',sep='\t', names=['eng','spa','drop_me'])

    eng_all = df['eng'].apply(clean_text)
    eng_all = eng_all.apply(start_end_tagger)

    spa_all = df['kor'].apply(clean_text)
    spa_all = spa_all.apply(start_end_tagger)

    eng_tensor, eng_tokenizer = tokenize(eng_all)
    spa_tensor, spa_tokenizer = tokenize(spa_all)

    eng_vocab_size = len(eng_tokenizer.word_index)+1
    spa_vocab_size = len(spa_tokenizer.word_index)+1

    eng_max_length = len(eng_tensor[0])
    spa_max_length = len(spa_tensor[0])

    eng_tensor1 = eng_tensor[:33000]
    spa_tensor1 = spa_tensor[:33000]

    # eng_tensor2 = eng_tensor[33000:66000]
    # spa_tensor2 = spa_tensor[33000:66000]

    # eng_tensor3 = eng_tensor[66000:99000]
    # spa_tensor3 = spa_tensor[66000:99000]

    # eng_tensor4 = eng_tensor[99000:]
    # spa_tensor4 = spa_tensor[99000:]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(eng_tensor1,spa_tensor1, test_size=0.2)
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(eng_tensor2,spa_tensor2, test_size=0.2)
    # X_train3, X_test3, y_train3, y_test3 = train_test_split(eng_tensor3,spa_tensor3, test_size=0.2)
    # X_train4, X_test4, y_train4, y_test4 = train_test_split(eng_tensor1,spa_tensor4, test_size=0.2)

    print('y encoding...')

    y_train1 = encode_output(y_train1, spa_vocab_size)
    y_test1 = encode_output(y_test1, spa_vocab_size)

    # y_train2 = encode_output(y_train2, spa_vocab_size)
    # y_test2 = encode_output(y_test2, spa_vocab_size)

    # y_train3 = encode_output(y_train3, spa_vocab_size)
    # y_test3 = encode_output(y_test3, spa_vocab_size)

    print('building model...')

    # model = define_model(eng_vocab_size, spa_vocab_size, eng_max_length, spa_max_length, 50)
    # model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['mae', 'acc'])

    log_dir="logs/fit_spa/"
    tensorboard_callback = TensorBoard(log_dir=log_dir_spa, histogram_freq=1, profile_batch=1000000)

    filename = 'spa_model_best.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True)

    print('fitting model...')
    model = load_model('spa_model.h5')

    model.fit(X_train1, y_train1, epochs=50, batch_size=100, validation_data=(X_test1, y_test1), callbacks=[checkpoint,tensorboard_callback], verbose=2)
    model.save('model.h5')
    # model.fit(X_train3, y_train3, epochs=10, batch_size=100, validation_data=(X_test3, y_test3), callbacks=[checkpoint,tensorboard_callback], verbose=2)

    print(predict_sequence(model,spa_tokenizer, X_test1[1000]))
