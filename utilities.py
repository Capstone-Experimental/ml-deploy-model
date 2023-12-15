import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import custom_object_scope
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
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
def load_predict_model_pretrain(texts):
    model_path = 'models/pretrain_sentiment.h5'  # Check if this path is correct
    model_pretrain = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    get_predictions = []
    for text in texts :
      predictions = model_pretrain.predict(text)
      print(predictions)
      get_predict = 'positif' if predictions[0] > 0.5 else 'negatif'
      get_predictions.append(get_predict)
      # print(padded_text)
    return get_predictions

# Pipeline Model
# Fungsi untuk melakukan tokenisasi menggunakan TensorFlow Tokenizer
class TokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def fit(self, X, y=None):
        # Tidak melakukan fit ulang pada tokenizer
        return self

    def transform(self, X):
        texts = []
        for text in X :
          new_text = text.lower()  # Ubah ke lowercase
          # new_text = re.sub(r'[^a-zA-Z0-9\s]', '', new_text)  # Hapus karakter khusus
          texts.append(new_text)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post', truncating='post'
        )
        return padded_sequences
# Load Models
def load_predict_model_pipeline(texts):
    file_path = 'models/pipeline_sentiment.pickle'
    loaded_model = load_model('models/sentimentv2_model.h5')
    print("Attempting to load from:", file_path)  # Debug print
    try:
        pipeline_sentiment = Pipeline([
            ('tokenizer', TokenizerTransformer(tokenizer, max_length)),
            ('prediction', loaded_model)
        ])
        predicts = pipeline_sentiment.predict(texts)
        return predicts
    except Exception as e:
        print("Error during loading:", e)  # Debug print
        return None  # Return None or handle the error accordingly

if __name__ == '__main__':
  text = ['belajar bahasa inggris yang baik']
  predictions = load_predict_model_pipeline(text)
  print(predictions)