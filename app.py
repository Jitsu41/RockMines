from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('sonar_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # expects list of 60 floats
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)[0]
    if pred == 'R':
        message = "Rock Ahead No Tension"
    else:
        message = "Mines Ahead Danger!!!"
    return jsonify({'prediction': message})

from flask import send_from_directory

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
