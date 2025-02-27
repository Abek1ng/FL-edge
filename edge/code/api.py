import os
import torch
import logging
from flask import Flask, request, jsonify
from model import MLPClassifier

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)

# Load Edge Model
model = MLPClassifier()
model_path = "./local_storage/edge_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    logger.info("Model loaded successfully from local storage.")
else:
    logger.info("No saved model found, using initial model.")

# Ensure model is in evaluation mode
model.eval()

def predict_anomaly(sensor_data):
    """Process incoming sensor data and predict anomaly."""
    try:
        features = torch.tensor([
            sensor_data["HR"],
            sensor_data["BT"],
            sensor_data["SpO2"],
            sensor_data["Age"],
            sensor_data["Gender"]
        ], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(features)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive sensor data and return anomaly prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        prediction = predict_anomaly(data)
        if prediction is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        return jsonify({"status": "success", "anomaly": prediction}), 200
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Edge Device API on Raspberry Pi...")
    app.run(host='0.0.0.0', port=5001, debug=False)
