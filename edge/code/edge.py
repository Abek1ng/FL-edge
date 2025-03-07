import os
import time
import threading
import random
import logging
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, request, jsonify
from tinydb import TinyDB, Query
from model import MLPClassifier

# ------------------ Configuration ------------------ #
SERVER_URL = "https://9953-87-255-216-86.ngrok-free.app"  # Federated server endpoint
DEVICE_ID = 1
LOCAL_DATA_PATH = f"./local_storage/local_data_{DEVICE_ID}.csv"
REPORT_DB_PATH = "./local_storage/reports.json"  # TinyDB storage
RETRAIN_THRESHOLD = 10
COOLDOWN_PERIOD = 30
NORMAL_FEEDBACK_PROB = 0.7
TRAINING_CHECK_INTERVAL = 10  # Seconds between retraining checks
FEEDBACK_WAIT_TIME = 30  # Time to wait for user feedback

# Ensure local storage exists
os.makedirs("./local_storage", exist_ok=True)
if not os.path.exists(LOCAL_DATA_PATH):
    pd.DataFrame(columns=["HR", "BT", "SpO2", "Age", "Gender", "Outcome"]).to_csv(LOCAL_DATA_PATH, index=False)

# Initialize TinyDB for logging reports
db = TinyDB(REPORT_DB_PATH)

# ------------------ Logging Setup ------------------ #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------ Flask App ------------------ #
app = Flask(__name__)

# ------------------ Model Initialization ------------------ #
model = MLPClassifier()
model_path = "./local_storage/edge_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    logger.info("Model loaded successfully from local storage.")
else:
    logger.warning("No saved local model found, using initial model.")

model.eval()  # Start in eval mode

# Store pending anomalies waiting for feedback
pending_feedback = {}


# ------------------ Utility Functions ------------------ #
def state_dict_to_json(state_dict):
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def json_to_state_dict(json_dict):
    return {k: torch.tensor(v) for k, v in json_dict.items()}

def fetch_global_model():
    """Fetch latest global model from the federated server."""
    try:
        response = requests.get(f"{SERVER_URL}/model")
        if response.status_code == 200:
            global_state = json_to_state_dict(response.json())
            model.load_state_dict(global_state, strict=False)
            logger.info("Global model updated locally.")
        else:
            logger.warning("Error fetching global model: %s", response.text)
    except Exception as e:
        logger.error("Exception during model download: %s", e)

def save_new_data(sensor_sample):
    """Append a sensor sample to the local CSV dataset."""
    try:
        df = pd.DataFrame([sensor_sample])
        df.to_csv(LOCAL_DATA_PATH, mode="a", header=False, index=False)
        db.insert(sensor_sample)  # Save to TinyDB for logging
        logger.info("Sample saved: %s", sensor_sample)
    except Exception as e:
        logger.error("Error saving new data: %s", e)

def send_model_update(local_model):
    """Send updated local model params to the federated server."""
    try:
        current_state = state_dict_to_json(local_model.state_dict())
        response = requests.post(f"{SERVER_URL}/update", json=current_state)
        logger.info("Server response: %s", response.json())
    except Exception as e:
        logger.error("Error sending update: %s", e)

def train_local_model():
    """Retrain the local model when new labeled data is available."""
    try:
        df_local = pd.read_csv(LOCAL_DATA_PATH)
        if len(df_local) < RETRAIN_THRESHOLD:
            return  # Not enough data to train

        logger.info("Retraining local model - data threshold reached.")
        X = df_local[["HR", "BT", "SpO2", "Age", "Gender"]].values.astype(np.float32)
        y = df_local["Outcome"].values.astype(np.int64)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(3):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        model.eval()
        logger.info("Retraining completed.")

        # Send update after retraining
        send_model_update(model)

    except Exception as e:
        logger.error("Error training local model: %s", e)


# ------------------ Flask Routes ------------------ #
@app.route('/send_data', methods=['POST'])
def sensor_data():
    """
    Receives sensor data in JSON format, predicts anomaly, and waits for user feedback if needed.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Extract features
        features = [
            data.get("HR", 0),
            data.get("BT", 0),
            data.get("SpO2", 0),
            data.get("Age", 0),
            data.get("Gender", 0)
        ]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predict anomaly
        model.eval()
        with torch.no_grad():
            output = model(x)
            prediction = torch.argmax(output, dim=1).item()  # 0 or 1

        # If no anomaly, store it immediately
        if prediction == 0:
            data["Outcome"] = 0
            save_new_data(data)
            return jsonify({"status": "success", "anomaly": 0}), 200

        # Store anomaly in pending_feedback and wait for user feedback
        anomaly_id = str(time.time())  # Unique anomaly ID
        pending_feedback[anomaly_id] = data
        logger.info(f"Anomaly detected! Waiting for feedback (ID: {anomaly_id})")

        # Wait for feedback (non-blocking)
        threading.Thread(target=wait_for_feedback, args=(anomaly_id,)).start()

        return jsonify({"status": "pending_feedback", "anomaly_id": anomaly_id}), 202

    except Exception as e:
        logger.error("API Error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/feedback', methods=['POST'])
def receive_feedback():
    """
    Receives user feedback in JSON format:
    {
        "anomaly_id": str,
        "real_label": int (0 or 1)
    }
    """
    feedback = request.get_json()
    anomaly_id = feedback.get("anomaly_id")
    real_label = feedback.get("real_label")

    if anomaly_id in pending_feedback:
        data = pending_feedback.pop(anomaly_id)
        data["Outcome"] = real_label
        save_new_data(data)
        return jsonify({"message": "Feedback received and stored."}), 200
    else:
        return jsonify({"error": "Invalid anomaly ID"}), 400


# ------------------ Background Thread to Wait for Feedback ------------------ #
def wait_for_feedback(anomaly_id):
    """Waits for feedback for 30 seconds. If no feedback is received, assume outcome = 1."""
    start_time = time.time()
    while time.time() - start_time < FEEDBACK_WAIT_TIME:
        if anomaly_id not in pending_feedback:  # Feedback received
            return
        time.sleep(1)

    # No feedback received, assume anomaly was real
    if anomaly_id in pending_feedback:
        logger.warning(f"No feedback received for anomaly ID {anomaly_id}. Assuming Outcome = 1.")
        data = pending_feedback.pop(anomaly_id)
        data["Outcome"] = 1
        save_new_data(data)


# ------------------ Main Entry Point ------------------ #
if __name__ == "__main__":
    fetch_global_model()
    threading.Thread(target=train_local_model, daemon=True).start()
    logger.info("Edge Device API starting on 0.0.0.0:5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
