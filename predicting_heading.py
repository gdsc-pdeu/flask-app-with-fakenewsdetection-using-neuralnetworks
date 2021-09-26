from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
tokenizer=pickle.load(open('flaskapp_ibmzdatathon/heading_tokenizer_l.pkl','rb'))
new_models=keras.models.load_model("flaskapp_ibmzdatathon/fakenews_temp")
def predict(sentence):
    
    # sentence=[
    #     "granny starting to fear spiders in the garden might be real",
    #     "the weather today is bright and sunny",
    #     "Progressive Couple Thrilled With Latest Mandates",
    #     "'Biden won US election against Trump in fair ways,' Republican-funded review claims"
    # ]
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"

    sequences=tokenizer.texts_to_sequences(sentence)
    padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

    print(new_models.predict(padded))
    x=new_models.predict(padded)

    [print(float("{:.8f}".format(i[0]))) for i in x]


    print("Hello World")
    return [float("{:.8f}".format(i[0])) for i in x]
    return "True"

print(predict(["Hello World"]))