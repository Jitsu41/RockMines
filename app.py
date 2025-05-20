from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='templates')
CORS(app)

# Load your trained model
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

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
