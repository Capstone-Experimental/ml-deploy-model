# Import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify
from utilities import load_predict_model

# Create an instance of the Flask class
app = Flask(__name__)

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
    predictions = load_predict_model(sample)
    try : 
        result = jsonify({'sentiment': predictions[0]})
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)