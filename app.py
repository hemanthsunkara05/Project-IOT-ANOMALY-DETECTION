from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# File to store incoming sensor data
DATA_FILE = 'sensor_data.csv'

# Create CSV if it doesn't exist
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["timestamp", "sensor_type", "value"])
    df.to_csv(DATA_FILE, index=False)

@app.route('/')
def home():
    return "✅ Flask IoT Server Running!"

# Endpoint to receive data from ESP32
@app.route('/upload-data', methods=['POST'])
def upload_data():
    try:
        data = request.get_json()

        # Extract values
        sensor_type = data['sensor_type']
        value = float(data['value'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to CSV
        df = pd.read_csv(DATA_FILE)
        new_row = pd.DataFrame([[timestamp, sensor_type, value]], columns=["timestamp", "sensor_type", "value"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        return jsonify({"status": "success", "message": "Data saved"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)