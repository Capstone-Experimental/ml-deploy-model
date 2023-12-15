# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
from threading import Thread
from utilities import load_predict_model,load_predict_model_pretrain
import csv

# Create an instance of the Flask class
app = Flask(__name__)

# Save new data
def log_to_csv(sample, sentiment):
    class_sentiment = {'positif' : 1, 'negatif':0}
    with open('dataset/data_sentiment_train.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['text', 'sentiment'])
        writer.writerow({'text': sample[0], 'sentiment': class_sentiment[sentiment] })

# Register a route
@app.get('/')
def home():
    text = ""
    if request.method == 'POST':
        text = request.form.get('text-content')
    return render_template("index.html", text=text)

# Predict Texts Sentiment
@app.post('/predict')
def predict():
    data = request.json
    try :
        sample = data['text']
    except KeyError:
        return jsonify({'error' : 'No text sent'})
    sample = [[sample]]

    # Make Prediction
    predictions = load_predict_model_pretrain(sample)
    predicted_sentiment = predictions[0]

    try : 
        result = jsonify({'sentiment': predicted_sentiment})
        # Start a thread for logging in the background
        thread = Thread(target=log_to_csv, args=(sample[0], predicted_sentiment))
        thread.start()
    except TypeError as e:
        result = jsonify({'error': str(e)})

    return result

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)