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

    padded_tensor = pad_sequences(tensor, maxlen=max_length(tensor), padding='post')

    return padded_tensor, lang_tokenizer


def preprocess(language):
    language = language.apply(clean_text)
    
    language = language.apply(start_end_tagger)
    
    return language

def preprocess_sentence(sentence):
    sentence = clean_text(sentence)
    sentence = start_end_tagger(sentence)
    return sentence


def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_object = SparseCategoricalCrossentropy()
    loss_ = loss_object(real,pred)
    
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)





def evaluate(sentence,max_length_targ,max_length_inp,encoder_model,decoder_model,input_lang_tokenizer,target_lang_tokenizer,units):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)
    
    input_sentence = [input_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    input_sentence = pad_sequences([input_sentence],maxlen=max_length_inp,padding='post')
    
    print(input_sentence)
    input_sentence_tensor = tf.convert_to_tensor(input_sentence)
    result = ''
    
    print(input_sentence_tensor)
    hidden = [tf.zeros((1, units))]
    enc_output, enc_hidden = encoder_model(input_sentence_tensor, hidden)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<start>']],0)
    
    for t in range(max_length_targ):
        pred, dec_hidden, attention_weights = decoder_model(dec_input, dec_hidden, enc_output)
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        
        pred_id = tf.argmax(pred[0]).numpy()
        
        result += target_lang_tokenizer.index_word[pred_id] + ' '
        
        if target_lang_tokenizer.index_word[pred_id] == '<end>':
            return result, sentence, attention_plot
        
        dec_input = tf.expand_dims([pred_id],0)
    return result, sentence, attention_plot


def plot_attention(attention, sentence, pred):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict={'fontsize':14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict)
    ax.set_yticklabels([''] + pred, fontdict=fontdict)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()


# @tf.function
def train_step(inp, targ, enc_hidden,encoder_model,decoder_model, target_lang_tokenizer,batch_size):
    optimizer = Adam()
    loss = 0
    
    with tf.GradientTape() as tape:
        enc_output, enc_hidden_state = encoder_model(inp, enc_hidden)
        
        dec_hidden_state = enc_hidden_state
        
        dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<start>']] * batch_size, 1)
        
        for t in range(1, targ.shape[1]):
            pred, dec_hidden, _ = decoder_model(dec_input, dec_hidden_state, enc_output)
            loss += loss_function(targ[:,t],pred)
            dec_input = tf.expand_dims(targ[:,t],1)
            
    batch_loss = (loss/int(targ.shape[1]))
    variables = encoder_model.trainable_variables + decoder_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients,variables))
    
    return batch_loss