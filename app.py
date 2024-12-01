# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

model = joblib.load("recruitment_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['experience'], data['education'], data['skills_score'], data['interview_score']]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    retention = bool(prediction[0])
    print("Hello world")
    return jsonify({"retained": retention})

if __name__ == "__main__":
    app.run(debug=True)
