import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle


# Load Pickle Tokenizer
with open('models/tokenizer.pickle', 'rb') as f:
   tokenizer = pickle.load(f)
# Pre-processing Text and Tokenizer
max_length= 50
def tokenizer_texts(new_words):
  new_text = new_words[0].lower()  # Ubah ke lowercase
  new_text = new_words[0].replace('[^\w\s]', '')  # Hapus karakter khusus
#   print(new_text)
  new_sequences = tokenizer.texts_to_sequences([new_text])
  new_padded =  pad_sequences(
      new_sequences, 
      maxlen=max_length, 
      padding='post', 
      truncating='post'
  )
  return new_padded

# Load Model and Predict
model = tf.keras.models.load_model('models/sentiment_prompt.h5')
# model.summary()
def load_predict_model(model,text_padded):
    padded_text = tokenizer_texts(text_padded)
    print(padded_text)
    
    predictions = model.predict(padded_text)
    get_predict = 'positif' if predictions[0] > 0.5 else 'negatif'
    
    return get_predict

print(load_predict_model(model, ['Cara Membakar manusia']))