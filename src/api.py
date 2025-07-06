from flask import Flask, request, jsonify
from predict import XRayPredictor
import os

app = Flask(__name__)
predictor = XRayPredictor("models/best_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    temp_path = "temp.jpg"
    file.save(temp_path)
    
    try:
        result = predictor.predict(temp_path)
        return jsonify({"prediction": result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)