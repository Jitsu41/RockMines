from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('sonar_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        arr = np.array(data).reshape(1, -1)
        pred = model.predict(arr)[0]

        message = "Rock Ahead No Tension" if pred == 'R' else "Mines Ahead Danger!!!"
        return jsonify({'prediction': message})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
