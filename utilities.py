import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
# import tensorflow_hub as hub
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
# import pickle

# Memuat tokenizer dari direktori lokal
max_length= 50
loaded_tokenizer = BertTokenizer.from_pretrained("tokenizer_indobert")
def tokenizerTexts(sentences):
    token_predict = loaded_tokenizer(sentences, padding="max_length", truncation=True, max_length=max_length)
    return token_predict.input_ids

# Load Model
# Menentukan jalur (path) ke model TensorFlow Lite yang akan digunakan.
interpreter = tf.lite.Interpreter(model_path='models/sentiment_quantized_model.tflite')
# Melakukan alokasi memori untuk interpreter.
interpreter.allocate_tensors()

# Predict
def load_predict_quantitazing(text):
    numerical_representations = tokenizerTexts(text)
    for i, sentence_representations in enumerate(numerical_representations):
        # Memasukkan data uji ke dalam input tensor model
        input_details = interpreter.get_input_details()
        input_text = np.array(sentence_representations).astype(np.float32)
        input_text = input_text.reshape(1, -1)
        # print(input_text)
        # input_details
        interpreter.set_tensor(input_details[0]['index'], input_text)

        # Melakukan inferensi
        interpreter.invoke()

        # Mendapatkan hasil prediksi dari output tensor model
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predict = "positif" if output_data > 0.5 else "negatif"
    return [predict]

# Test
if __name__ == '__main__':
    get_p = load_predict_quantitazing(['"membuat miras oplosan yang tidak membunuh'])
    print(get_p)
#     # Negatif
#     test_sentences = ["mendaki gunung dengan hati-hati", "belajar bermain sepeda sampai hebat", "Menjual sabu tanpa ketahuan", "membuat miras oplosan yang tidak membunuh"]

#     # Menggunakan fungsi tokenizerTexts untuk mendapatkan representasi numerik dari kalimat-kalimat tersebut
#     numerical_representations = tokenizerTexts(test_sentences)

#     for i, sentence_representations in enumerate(numerical_representations):
#         # Memasukkan data uji ke dalam input tensor model
#         input_details = interpreter.get_input_details()
#         input_text = np.array(sentence_representations).astype(np.float32)
#         input_text = input_text.reshape(1, -1)
#         # print(input_text)
#         # input_details
#         interpreter.set_tensor(input_details[0]['index'], input_text)

#         # Melakukan inferensi
#         interpreter.invoke()

#         # Mendapatkan hasil prediksi dari output tensor model
#         output_details = interpreter.get_output_details()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         predict = "positif" if output_data > 0.5 else "negatif"
        
#         # Menampilkan hasil prediksi
#         print(f"Kalimat {i+1}: {test_sentences[i]}")
#         print("Hasil prediksi:")
#         print(f'Prediksi = {output_data}, Hasil = {predict}')
#   test_sentences = ["mendaki gunung dengan hati-hati", "belajar bermain sepeda sampai hebat", "Menjual sabu tanpa ketahuan", "membuat miras oplosan yang tidak membunuh"]
#   predictions = load_predict_model_pipeline(text)
#   print(predictions)