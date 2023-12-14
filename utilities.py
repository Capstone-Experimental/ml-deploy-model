import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences
from keras.utils import custom_object_scope
import pickle


# Load Pickle Tokenizer
with open('models/tokenizer.pickle', 'rb') as f:
   tokenizer = pickle.load(f)

# Pre-processing Text and Tokenizer
max_length= 50
def tokenizer_texts(new_words):
  new_text = new_words[0].lower()  # Ubah ke lowercase
  new_text = new_words[0].replace('[^\w\s]', '')  # Hapus karakter khusus
# print(new_text)
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
def load_predict_model(texts):
    get_predictions = []
    for text in texts :
      padded_text = tokenizer_texts(text)
      predictions = model.predict(padded_text)
      print(predictions)
      get_predict = 'positif' if predictions[0] > 0.5 else 'negatif'
      get_predictions.append(get_predict)
      # print(padded_text)
    return get_predictions

# Pre Train Model
model_path = 'models/pretrain_sentiment.keras'  # Check if this path is correct
def load_predict_model_pretrain(texts):
    model_pretrain = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    predictions = model_pretrain.predict(texts)
    return predictions

text = [['Cara membunuh manusia']]
predictions = load_predict_model_pretrain(text)
print(predictions)