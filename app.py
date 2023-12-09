# Import the Flask class from the flask module
from flask import Flask, render_template, request

# Create an instance of the Flask class
app = Flask(__name__)

# Register a route
@app.route('/', methods=['GET', 'POST'])
def home():
    text = ""
    if request.method == 'POST':
        text = request.form.get('text-content')
    return render_template("index.html", text=text)

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)