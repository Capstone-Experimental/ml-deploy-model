import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer


dataset = pd.read_csv('dataset/data_sentimenv2.csv')

# Cleaning Data
dataset['text'] = dataset['text'].str.lower()  # Ubah ke lowercase
dataset['text'] = dataset['text'].str.replace('[^\w\s]', '')  # Hapus karakter khusus

sentences = dataset['text'].tolist()

tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(sentences)

print('Success')