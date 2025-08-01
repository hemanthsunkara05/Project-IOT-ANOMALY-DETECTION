from flask import Flask, request, jsonify, render_template, send_file
import csv
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd  # MISSING IMPORT FIXED

model = joblib.load("model/isolation_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
app = Flask(__name__)

CSV_FILE = 'sensor_data.csv'

# POST route to receive sensor data
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Add timestamp if not provided
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sensor_id = data.get('sensor_id', 'default_sensor')
    sensor_type = data.get('sensor_type', 'default_type')
    value = data.get('value', None)

    # Validate value
    try:
        sensor_value = float(value)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid value for sensor'}), 400

    # Append to CSV (ensure consistent column order)
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['sensor_id', 'sensor_type', 'value', 'timestamp'])  # header
        writer.writerow([sensor_id, sensor_type, sensor_value, data['timestamp']])

    # Prepare data for anomaly detection
    # Create a one-row DataFrame with all expected features (model.feature_names_in_)
    sample_dict = {ft: 0 for ft in model.feature_names_in_}
    if sensor_type in sample_dict:
        sample_dict[sensor_type] = sensor_value
    sample_full = pd.DataFrame([sample_dict])

    # Scale and predict
    X_test = scaler.transform(sample_full)
    prediction = model.predict(X_test)  # -1 = anomaly, 1 = normal
    is_anomaly = int(prediction[0] == -1)

    return jsonify({'message': 'Data received and saved', 'is_anomaly': is_anomaly}), 200

# GET route to return data as JSON
@app.route('/data', methods=['GET'])
def serve_data():
    result = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                result.append({
                    'sensor_id': row.get('sensor_id', ''),
                    'sensor_type': row.get('sensor_type', ''),
                    'value': row.get('value', ''),
                    'timestamp': row.get('timestamp', '')
                })
    return jsonify(result)

# Route to render dashboard HTML
@app.route('/dashboard')
def dashboard():
    timestamps = []
    values = []
    try:
        with open(CSV_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestamps.append(row.get('timestamp', ''))
                try:
                    values.append(float(row.get('value', 0.0)))
                except ValueError:
                    values.append(0.0)
    except FileNotFoundError:
        timestamps, values = [], []

    return render_template('dashboard.html', timestamps=timestamps, values=values)

# Route to download the CSV file
@app.route('/download')
def download_file():
    if not os.path.exists(CSV_FILE):
        return "No data file found", 404
    return send_file(CSV_FILE, as_attachment=True)

# Route to return data grouped by sensor_type as JSON
@app.route('/data_json')
def data_json():
    if not os.path.exists(CSV_FILE):
        return jsonify({})
    df = pd.read_csv(CSV_FILE)
    data = {}
    if 'sensor_type' in df.columns:
        for sensor_type in df['sensor_type'].unique():
            sensor_df = df[df['sensor_type'] == sensor_type]
            data[sensor_type] = {
                "timestamps": sensor_df['timestamp'].tolist(),
                "values": sensor_df['value'].tolist()
            }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)