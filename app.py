from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load your trained model once at startup
model = joblib.load('sonar_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # expects list of 60 floats
        arr = np.array(data).reshape(1, -1)
        pred = model.predict(arr)[0]

        if pred == 'R':
            message = "Rock Ahead No Tension"
        else:
            message = "Mines Ahead Danger!!!"

        return jsonify({'prediction': message})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use the PORT environment variable for deployment platforms like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
