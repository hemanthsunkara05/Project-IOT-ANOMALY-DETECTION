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
FEATURES = ['mq5_01']  # Replace with actual sensor types you used during training

CSV_FILE = 'sensor_data.csv'

# POST route to receive sensor data and run model prediction
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    sensor_type = data.get('sensor_type')
    value = data.get('value')

    # Save to sensor_data.csv
    df = pd.DataFrame([{
        'timestamp': timestamp,
        'sensor_type': sensor_type,
        'value': value
    }])
    df.to_csv('sensor_data.csv', mode='a', header=not os.path.exists('sensor_data.csv'), index=False)

    # === Load model & scaler ===
    try:
        model = joblib.load('model/isolation_forest_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
    except Exception as e:
        return jsonify({'error': f'Failed to load model/scaler: {str(e)}'})

    # === Prepare input in pivoted format ===
    try:
        pivot_data = pd.read_csv('sensor_data.csv')
        pivot_data = pivot_data[pd.to_numeric(pivot_data['value'], errors='coerce').notnull()]
        pivot_data['value'] = pivot_data['value'].astype(float)
        pivot_df = pivot_data.pivot_table(index='timestamp', columns='sensor_type', values='value', aggfunc='mean').fillna(0)
    except Exception as e:
        return jsonify({'error': f'Failed to prepare data for prediction: {str(e)}'})

    # Predict only latest row
    try:
        latest_data = pivot_df.iloc[[-1]]
        latest_scaled = scaler.transform(latest_data)
        prediction = model.predict(latest_scaled)[0]  # -1 = anomaly, 1 = normal
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

    # === If anomaly, log to anomalies.csv ===
    if prediction == -1:
        anomaly_record = {
            'timestamp': timestamp,
            'sensor_type': sensor_type,
            'value': value
        }
        pd.DataFrame([anomaly_record]).to_csv('anomalies.csv', mode='a', header=not os.path.exists('anomalies.csv'), index=False)

    return jsonify({
        'anomaly': int(prediction == -1),
        'message': 'Data received and prediction made.'
    })

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
@app.route('/dashboard-data')
def dashboard_data():
    data = []
    with open('sensor_data.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 4:  # skip rows without prediction
                continue
            timestamp, sensor_type, value, prediction = row
            data.append({
                'timestamp': timestamp,
                'sensor_type': sensor_type,
                'value': float(value),
                'anomaly': int(prediction)
            })
    return jsonify(data)


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