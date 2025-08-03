from flask import Flask, request, jsonify, render_template, send_file
import csv
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd  # MISSING IMPORT FIXED
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle


model = joblib.load("model/isolation_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
app = Flask(__name__)
FEATURES = ['mq5_01']  # Replace with actual sensor types you used during training

CSV_FILE = 'sensor_data.csv'
@app.route('/')
def home():
    return redirect("/dashboard")  # optional redirect to dashboard

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")
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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and scaler
        with open("model/scaler.pkl", "rb") as s:
            scaler = pickle.load(s)
        with open("model/isolation_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Get JSON input
        data = request.get_json()
        sensor_type = data.get("sensor_type")
        value = data.get("value")

        if value is None or sensor_type is None:
            return jsonify({"error": "Missing 'sensor_type' or 'value' in input JSON"}), 400

        # Build full feature vector
        features = ['mq5_01']  # 👈 Make sure this matches training
        feature_vector = {ft: 0.0 for ft in features}
        if sensor_type in features:
            feature_vector[sensor_type] = float(value)

        # Convert to DataFrame
        df = pd.DataFrame([feature_vector])

        # Scale and predict
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        status = "Anomaly" if prediction == -1 else "Normal"

        # ✅ Log this prediction to CSV
        df['anomaly'] = prediction
        df['sensor_id'] = data.get("sensor_id", "unknown")
        df['sensor_type'] = sensor_type
        df['value'] = value
        df['timestamp'] = datetime.now().isoformat()

        df.to_csv("sensor_data.csv", mode='a', index=False, header=not os.path.exists("sensor_data.csv"))

        return jsonify({
            "sensor_id": df['sensor_id'].values[0],
            "sensor_type": sensor_type,
            "value": value,
            "prediction": status
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
@app.route('/predictions')
def get_predictions():
    try:
        df = pd.read_csv("predictions.csv")
        return df.to_json(orient='records')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    try:
        df = pd.read_csv('sensor_data.csv')

        if 'timestamp' not in df.columns:
            return jsonify([])

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True)

        print("Dashboard Data Preview:\n", df.tail(5))  # ✅ Debug

        # Build response
        result = []
        for _, row in df.iterrows():
            result.append({
                'sensor_type': row['sensor_type'],
                'value': row['value'],
                'timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'anomaly': row['anomaly']
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



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

from flask import Flask, render_template



if __name__ == "__main__":
    app.run(debug=True)