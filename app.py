from flask import Flask, render_template, request
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

from helpme_l2_drop import *

app = Flask(__name__)




@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict/',methods=['POST'])
def predict():
	
    sentence=[x for x in request.form.values()][0]
    test_sentence1 = translate(sentence, eng_tokenizer,eng_max_length)
    prediction = predict_sequence(model,kor_tokenizer,test_sentence1)
    return render_template('index.html', prediction_text='{}'.format(prediction[8:]))
    

if __name__ == "__main__":
    model = load_model('model_l2_drop.h5')

    df_all = pd.read_csv('data/final_df_fix.txt',sep='\t')
    eng_all = preprocess(df_all['eng'])
    kor_all = preprocess(df_all['kor'])

    eng_tensor, eng_tokenizer = tokenize(eng_all)
    kor_tensor, kor_tokenizer = tokenize(kor_all)

    eng_vocab_size = len(eng_tokenizer.word_index)+1
    kor_vocab_size = len(kor_tokenizer.word_index)+1

    eng_max_length = len(eng_tensor[0])
    kor_max_length = len(kor_tensor[0])
    app.run()