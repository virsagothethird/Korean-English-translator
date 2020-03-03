import pandas as pd
import numpy as np
# from langdetect import detect
import re
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import datetime
import gc

plt.style.use('seaborn')

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
    """
    Clean text by removing unnecessary characters and altering the format of words.

    Parameters
    ----------
    text : pandas series

    Returns
    -------
    pandas series with cleaned text
    """
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


def start_end_tagger(sentence):
    """
    Adds a <start> and <end> tag to each sentence.

    Parameters
    ----------
    sentence : str

    Returns
    -------
    str
    """
    
    start_tag = "<start> "
    end_tag = " <end>"
    final_target = start_tag + sentence + end_tag
    return final_target


def max_length(sequence):
    """
    Returns the max length of a sequence.

    Parameters
    ----------
    sequence : array

    Returns
    -------
    int
    """
    return max(len(t) for t in sequence)


def tokenize(clean_text):
    """
    Transforms cleaned text to numerical values

    Parameters
    ----------
    clean_text : Pandas series

    Returns
    -------
    Array, tokenizer object
    """

    lang_tokenizer = Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(clean_text)

    tensor = lang_tokenizer.texts_to_sequences(clean_text)

    padded_tensor = pad_sequences(tensor, maxlen=max_length(tensor), padding='post',truncating='pre')

    return padded_tensor, lang_tokenizer


def preprocess(language):
    """
    Applies the clean_text and start_end_tagger functions to pandas series

    Parameters
    ----------
    language : Pandas series

    Returns
    -------
    Pandas series
    """
    language = language.apply(clean_text)
    
    language = language.apply(start_end_tagger)
    
    return language

def preprocess_sentence(sentence):
    """
    Applies the clean_text and start_end_tagger functions to a string

    Parameters
    ----------
    sentence : str

    Returns
    -------
    str
    """
    
    sentence = clean_text(sentence)
    sentence = start_end_tagger(sentence)
    return sentence

def encode_output(sequences, vocab_size):
    """
    One Hot Encodes the sequences

    Parameters
    ----------
    sequences : 2d array
    vocab_size : int, length of tokenizer object word index +1

    Returns
    -------
    3d array
    """

    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def define_model(inp_vocab_size, tar_vocab_size, inp_timesteps, tar_timesteps, n_units):
    """
    Creates a Tensorflow Sequential Model

    Parameters
    ----------
    inp_vocab_size : int, length of input tokenizer object word index +1
    tar_vocab_size : int, length of output tokenizer object word index +1
    inp_timesteps : int, input sequence max length
    tar_timesteps : int, target sequence max length
    n_units : int

    Returns
    -------
    Tensorflow Sequential Model
    """

    model = Sequential()
    model.add(Embedding(inp_vocab_size, n_units, input_length=inp_timesteps, mask_zero=True))
    model.add(LSTM(n_units,recurrent_regularizer='l2', dropout=0.15))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True,recurrent_regularizer='l2', dropout=0.15))
    model.add(TimeDistributed(Dense(tar_vocab_size, activation='softmax')))
    return model


def word_for_id(index, tokenizer):
    """
    Returns word from vocabulary at index

    Parameters
    ----------
    index : int, index
    tokenizer : tokenizer object

    Returns
    -------
    str
    """

    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


def decode_input(input_sequences,index,tokenizer):
    """
    Converts input sequence back to string

    Parameters
    ----------
    input_sequences : 2d array
    index : int, index
    tokenizer : tokenizer object

    Returns
    -------
    str
    """

    sentence =[]
    for i in input_sequences[index]:
        if tokenizer.index_word[i]=='<end>':
            break

        sentence.append(tokenizer.index_word[i])
    return " ".join(sentence)


def encode_sentence(sentence,tokenizer,max_length):
    """
    Encodes string to numerical array

    Parameters
    ----------
    sentence : str
    tokenizer : tokenizer object
    max_length : int, max length of sentence

    Returns
    -------
    array
    """
    
    sentence = preprocess_sentence(sentence)
    
    input_sentence = [tokenizer.word_index[i] for i in sentence.split(' ')]
    input_sentence = pad_sequences([input_sentence],maxlen=max_length,padding='post')
    # input_sentence_tensor = tf.convert_to_tensor(input_sentence)
    return input_sentence[0]


def predict_sequence(model, tokenizer, input_sequence):
    """
    Returns the predicted translation

    Parameters
    ----------
    model : trained model
    tokenizer : tokenizer object
    input_sequence : array, encoded input

    Returns
    -------
    str
    """

    source = input_sequence.reshape((1, input_sequence.shape[0]))
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word == '<end>':
            break
        target.append(word)
    return ' '.join(target)


def plot_metrics(history, metric, val_metric, num):
    """
    Plots the metric of model during training. Saves plot

    Parameters
    ----------
    history : history of model
    metric : str, 'loss' or 'acc'
    val_metric : str, 'val_loss' or 'val_acc'
    num : int

    Returns
    -------
    None
    """

    fig,ax = plt.subplots(figsize=(10,6))

    ax.plot(history.history[metric])
    ax.plot(history.history[val_metric])
    ax.set_title('Model '+metric)
    ax.set_ylabel(metric)
    ax.set_xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(f'{metric}_plot{num}.png')

def train_batches(input_set, target_set, tar_vocab_size, model, epochs, loops, checkpoint_object, tensorboard_object):
    """
    Batch loads data into model. Saves model

    Parameters
    ----------
    input_set : history of model
    target_set : str, 'loss' or 'acc'
    tar_vocab_size : str, 'val_loss' or 'val_acc'
    model : model object
    epochs : int, epochs to train
    loops : int, number of loops through data
    checkpoint_object = checkpoint object
    tensorboard_object = tensorboard object

    Returns
    -------
    None
    """

    for loop in range(loops):
        j=0
        for i in np.linspace(input_set.shape[0]/5,input_set.shape[0],5):

            k = int(i)
            inp_sample = input_set[j:k]
            tar_sample = target_set[j:k]

            X_train, X_test, y_train, y_test = train_test_split(inp_sample,tar_sample, test_size=.1)

            print("Encoding target...")

            y_train = encode_output(y_train, tar_vocab_size)
            y_test = encode_output(y_test, tar_vocab_size)

            gc.collect()

            print("Fitting model...")

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=400, validation_data=(X_test, y_test), callbacks=[checkpoint_object,tensorboard_object], verbose=2)

            #fig,ax = plt.subplots(figsize=(10,6))

            plot_metrics(history, 'loss', 'val_loss',j)
            plot_metrics(history, 'acc', 'val_acc',j)

            j=int(i)

            model.save('model_l2_drop.h5')



if __name__ == "__main__":

    print('starting...')

    df_all = pd.read_csv('final_df_fix.txt',sep='\t')

    eng_all = preprocess(df_all['eng'])
    kor_all = preprocess(df_all['kor'])

    eng_tensor, eng_tokenizer = tokenize(eng_all)
    kor_tensor, kor_tokenizer = tokenize(kor_all)

    eng_vocab_size = len(eng_tokenizer.word_index)+1
    kor_vocab_size = len(kor_tokenizer.word_index)+1

    eng_max_length = len(eng_tensor[0])
    kor_max_length = len(kor_tensor[0])

    log_dir="logs/fit_l2_drop/"
    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                        histogram_freq=1,
                                        profile_batch=1000000)

    filename = 'model_l2_drop_best.h5'
    checkpoint = ModelCheckpoint(filename,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True)

    # model = define_model(eng_vocab_size, kor_vocab_size, eng_max_length, kor_max_length, 50)
    # model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
    model = load_model('model_l2_drop.h5')

    train_batches(eng_tensor,
                    kor_tensor,
                    kor_vocab_size,
                    model,
                    50,
                    1,
                    checkpoint,
                    tensorboard_callback)

    sentence = "where am i supposed to sleep"
    test_sentence = encode_sentence(sentence, eng_tokenizer,eng_max_length)

    print(f"Sentence 1: {sentence}\n\nModel Translation: {predict_sequence(model,kor_tokenizer,test_sentence)}")

    print(f"Sentence 2: {decode_input(eng_tensor,1000,eng_tokenizer)}\n\nModel Translation: {predict_sequence(model,kor_tokenizer, eng_tensor[1000])}")

