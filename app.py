# Import the Flask class from the flask module
import os

from flask import Flask, request, jsonify
from threading import Thread
from utilities import load_predict_model
from gcs import append_data_to_csv_in_gcs
from dotenv import load_dotenv


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
load_dotenv()

app = Flask(__name__)

@app.get('/api/trigger')
def test():
    return jsonify({"message" : "trigger"})

@app.post('/api/sentiment')
def predict():
    data = request.json
    try :
        sample = data['prompt']
    except KeyError:
        return jsonify({'error' : 'No text sent hehe'})
    sample = [[sample]]

    predictions = load_predict_model(sample)
    str_sample = str(sample[0]).replace("'", "").replace("[", "").replace("]", "")
    predicted_sentiment = predictions[0]
    class_sentiment = {'positif' : 1, 'negatif':0}
    try : 
        # append_data_to_csv_in_gcs(
        #     os.getenv('BUCKET_NAME'),
        #     os.getenv('BUCKET_PATH'),
        #     prompt=str_sample,
        #     sentiment=class_sentiment[predicted_sentiment]
        # ) 
        result = jsonify({'sentiment': predicted_sentiment})
    except TypeError as e:
        result = jsonify({'error': str(e)})

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)