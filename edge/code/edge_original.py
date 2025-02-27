import time
import random
import threading
import requests
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import MLPClassifier

SERVER_URL = "https://f83c-185-48-148-175.ngrok-free.app"  # Adjust if necessary

# Helper functions for JSON serialization of state_dict
def state_dict_to_json(state_dict):
    json_dict = {}
    for key, tensor in state_dict.items():
        json_dict[key] = tensor.detach().cpu().numpy().tolist()
    return json_dict

def json_to_state_dict(json_dict):
    state_dict = {}
    for key, value in json_dict.items():
        state_dict[key] = torch.tensor(value)
    return state_dict

def download_global_model():
    """Download the global model parameters from the server."""
    try:
        response = requests.get(f"{SERVER_URL}/model")
        if response.status_code == 200:
            json_state = response.json()
            state_dict = json_to_state_dict(json_state)
            return state_dict
        else:
            print("Error downloading global model:", response.text)
    except Exception as e:
        print("Exception during model download:", e)
    return None

def assess_local_model(model, X, y):
    """Assess the local model on provided data."""
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X))
        preds = torch.argmax(outputs, dim=1).numpy()
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)
    return acc, report

def load_local_data(csv_path):
    """Load local CSV data containing HR, BT, SpO2, Age, Gender, and Outcome."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        return None, None
    X = df[['HR', 'BT', 'SpO2', 'Age', 'Gender']].values.astype(np.float32)
    y = df['Outcome'].values.astype(np.int64)
    return X, y

def compute_class_weights(y):
    """Compute class weights to handle data imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    num_classes = len(classes)
    weights = {}
    for cls, count in zip(classes, counts):
        weights[cls] = total / (num_classes * count)
    weight_tensor = torch.tensor([weights.get(0, 1.0), weights.get(1, 1.0)], dtype=torch.float32)
    return weight_tensor

def train_local_model(model, X, y, epochs=5, batch_size=16, learning_rate=0.01):
    """Train the model locally using weighted CrossEntropyLoss and assess performance."""
    model.train()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_weights = compute_class_weights(y)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
   
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataset)
        print(f"Local Training - Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    # Assess model performance on the entire training data
    acc, report = assess_local_model(model, X, y)
    print(f"Local Model Training Assessment - Accuracy: {acc:.4f}")
    print(report)
    return model

def send_model_update(model, device_id):
    """Serialize the model state and send it to the server."""
    try:
        state_dict = model.state_dict()
        json_state = state_dict_to_json(state_dict)
        response = requests.post(f"{SERVER_URL}/update", json=json_state)
        print(f"Device {device_id}: Server response:", response.json())
    except Exception as e:
        print(f"Device {device_id}: Error sending update:", e)

class EdgeDevice:
    def __init__(self, device_id, dataset_path, server_url, retrain_threshold=10, cooldown_period=30, normal_feedback_prob=0.7,):
        self.device_id = device_id
        self.dataset_path = dataset_path  # Simulates a pool of sensor data
        self.server_url = server_url
        self.retrain_threshold = retrain_threshold  # New samples needed before retraining
        self.cooldown_period = cooldown_period  # Minimum seconds between update transmissions
        self.last_update_time = 0
        self.normal_feedback_prob = normal_feedback_prob  # Probability to record a normal sample
        self.data_iter_num = 0
        # Local file to accumulate samples for training
        self.local_data_path = f"./local_storage/local_data_{device_id+1}.csv"
        if not os.path.exists("./local_storage"):
            os.makedirs("./local_storage")
        if not os.path.exists(self.local_data_path):
            pd.DataFrame(columns=["HR", "BT", "SpO2", "Age", "Gender", "Outcome"]).to_csv(self.local_data_path, index=False)
        
        # Load initial sensor data (simulate streaming)
        self.data = self.load_initial_data()
        
        # Initialize local model and fetch the global model
        self.model = MLPClassifier()
        self.fetch_global_model(force_update=True)
        
    def load_initial_data(self):
        """Load dataset used to simulate sensor streaming."""
        if not os.path.exists(self.dataset_path):
            print(f"Device {self.device_id}: Dataset not found at {self.dataset_path}.")
            return pd.DataFrame(columns=["HR", "BT", "SpO2", "Age", "Gender", "Outcome"])
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"Device {self.device_id}: Loaded {len(df)} sensor samples from {self.dataset_path}.")
            return df
        except Exception as e:
            print(f"Device {self.device_id}: Error loading dataset: {e}")
            return pd.DataFrame(columns=["HR", "BT", "SpO2", "Age", "Gender", "Outcome"])
    
    def fetch_global_model(self, force_update=False):
        """Fetch the latest global model from the server."""
        global_state = download_global_model()
        if global_state:
            self.model.load_state_dict(global_state)
            print(f"Device {self.device_id}: Global model loaded.")
        else:
            print(f"Device {self.device_id}: Failed to fetch global model; using local copy.")
    
    def get_sensor_data(self):
        """Simulate real-time sensor data by sampling one row from the dataset."""
        if self.data.empty:
            return None
        if self.data_iter_num == len(self.data):
            return None
        self.data_iter_num+=1
        return self.data.sample(n=1).iloc[0].to_dict()
    
    def detect_anomaly(self, sensor_data):
        """Predict outcome using the local model for a sensor sample."""
        feature_keys = ["HR", "BT", "SpO2", "Age", "Gender"]
        features = [sensor_data[key] for key in feature_keys]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            prediction = torch.argmax(outputs, dim=1).item()
        return prediction
    
    def save_new_data(self, new_sample):
        """Append a new sensor sample (with feedback) to local storage."""
        try:
            df = pd.DataFrame([new_sample])
            df.to_csv(self.local_data_path, mode="a", header=False, index=False)
            print(f"Device {self.device_id}: New sample saved to local storage.")
        except Exception as e:
            print(f"Device {self.device_id}: Error saving new data: {e}")
    
    def load_local_training_data(self):
        """Load accumulated local training data."""
        if os.path.exists(self.local_data_path):
            try:
                df = pd.read_csv(self.local_data_path)
                return df
            except Exception as e:
                print(f"Device {self.device_id}: Error loading local training data: {e}")
        return pd.DataFrame(columns=["HR", "BT", "SpO2", "Age", "Gender", "Outcome"])
    
    def train_local_model(self, epochs=5, batch_size=16, learning_rate=0.01):
        """Retrain the local model using accumulated local data."""
        df = self.load_local_training_data()
        if df.empty or df["Outcome"].nunique() < 2:
            print(f"Device {self.device_id}: Not enough training data diversity for retraining.")
            return
        print(f"Device {self.device_id}: Retraining local model on {len(df)} samples.")
        X = df[["HR", "BT", "SpO2", "Age", "Gender"]].values.astype(np.float32)
        y = df["Outcome"].values.astype(np.int64)
        self.model = train_local_model(self.model, X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    def send_model_update(self):
        """Send updated model parameters to the server if the cooldown period has passed."""
        # current_time = time.time()
        # if current_time - self.last_update_time < self.cooldown_period:
        #     print(f"Device {self.device_id}: Skipping update due to cooldown.")
        #     return
        try:
            state_dict = self.model.state_dict()
            json_state = state_dict_to_json(state_dict)
            response = requests.post(f"{self.server_url}/update", json=json_state)
            print(f"Device {self.device_id}: Server response:", response.json())
            # self.last_update_time = current_time
        except Exception as e:
            print(f"Device {self.device_id}: Error sending update: {e}")
    
    def run(self):
        """Main loop: simulate sensor streaming, real-time prediction, and periodic retraining with hybrid feedback."""
        while True:
            sensor_data = self.get_sensor_data()
            if sensor_data is None:
                print(f"Device {self.device_id}: No sensor data available. Exiting loop.")
                break
            
            predicted = self.detect_anomaly(sensor_data)
            print(f"Device {self.device_id}: Sensor sample predicted as {predicted} (Actual: {sensor_data.get('Outcome')})")
            
            # If anomaly is predicted, use real feedback (as already implemented)
            if predicted == 1:
                # In this case, we assume the feedback is derived from the real outcome
                user_feedback = sensor_data.get("Outcome", 1)
                sensor_data["Outcome"] = user_feedback
                print(f"Device {self.device_id}: Anomaly detected! Recording sample with outcome: {user_feedback}")
                self.save_new_data(sensor_data)
            else:
                # For normal predictions, record a sample with a certain probability
                if random.random() < self.normal_feedback_prob:
                    print(f"Device {self.device_id}: Normal sample recorded for training.")
                    self.save_new_data(sensor_data)
            
            # Check if enough samples have been collected for retraining
            df_local = self.load_local_training_data()
            if len(df_local) >= self.retrain_threshold:
                self.train_local_model(epochs=3)  # Retrain for a few epochs
                self.retrain_threshold += 10  # Increase threshold to simulate accumulation
                self.send_model_update()
            
            time.sleep(1)  # Simulate sensor delay (1 sample per second)

def simulate_edge_device(device_id, dataset_path, server_url):
    device = EdgeDevice(device_id, dataset_path, server_url)
    device.run()

if __name__ == "__main__":
    import sys
    # Use provided CSV file paths or default ones for simulation
    csv_files = sys.argv[1:]
    if not csv_files:
        # csv_files = ["../data/client_data_1.csv"]
        csv_files = ["../data/client_data_8_resampled.csv","../data/client_data_9_resampled.csv","../data/client_data_10_resampled.csv"]
    threads = []
    for i, csv_file in enumerate(csv_files):
        t = threading.Thread(target=simulate_edge_device, args=(i, csv_file, SERVER_URL))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
