from flask import Flask, redirect, request, jsonify, render_template, send_file, Response, abort
import os
import pandas as pd
import joblib
from datetime import datetime
import pickle

app = Flask(__name__)

FEATURE_MAP = {
    "MQ-5": "mq5_01"
}

CSV_FILE = 'sensor_data.csv'
MODEL_DIR = 'model/'

# Load model and scaler once on startup
try:
    model = joblib.load(MODEL_DIR + "isolation_forest_model.pkl")
    scaler = joblib.load(MODEL_DIR + "scaler.pkl")
except Exception:
    model = None
    scaler = None

@app.route('/')
def home():
    return redirect("/dashboard")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    raw_sensor_type = data.get('sensor_type')
    sensor_type = FEATURE_MAP.get(raw_sensor_type)
    if sensor_type is None:
        return jsonify({'error': f'Unsupported sensor_type: {raw_sensor_type}'}), 400

    value = data.get('value')
    sensor_id = data.get('sensor_id', 'unknown')

    if value is None:
        return jsonify({'error': "Missing 'value'"}), 400

    global model, scaler
    if model is None or scaler is None:
        try:
            model = joblib.load(MODEL_DIR + 'isolation_forest_model.pkl')
            scaler = joblib.load(MODEL_DIR + 'scaler.pkl')
        except Exception as e:
            return jsonify({'error': f'Failed to load model/scaler: {str(e)}'}), 500

    if os.path.exists(CSV_FILE):
        all_data = pd.read_csv(CSV_FILE)
    else:
        columns = ['timestamp', 'sensor_id', 'sensor_type', 'value', 'anomaly']
        all_data = pd.DataFrame(columns=columns)

    new_row = {
        'timestamp': timestamp,
        'sensor_id': sensor_id,
        'sensor_type': sensor_type,
        'value': float(value),
        'anomaly': 0
    }
    all_data = pd.concat([all_data, pd.DataFrame([new_row])], ignore_index=True)

    try:
        pivot_df = all_data.pivot_table(index='timestamp', columns='sensor_type', values='value', aggfunc='mean').fillna(0)
        latest_data = pivot_df.iloc[[-1]]
        latest_scaled = scaler.transform(latest_data)
        prediction = int(model.predict(latest_scaled)[0] == -1)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    all_data.at[all_data.index[-1], 'anomaly'] = prediction
    all_data.to_csv(CSV_FILE, index=False)

    return jsonify({
        'anomaly': prediction,
        'message': 'Data received and prediction made.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        with open(MODEL_DIR + "scaler.pkl", "rb") as s:
            scaler = pickle.load(s)
        with open(MODEL_DIR + "isolation_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        data = request.get_json()
        sensor_type = data.get("sensor_type")
        value = data.get("value")
        sensor_id = data.get("sensor_id", "unknown")

        if sensor_type is None or value is None:
            return jsonify({"error": "Missing 'sensor_type' or 'value' in input JSON"}), 400

        features = ['mq5_01']
        feature_vector = {ft: 0.0 for ft in features}
        # Map input sensor_type to feature name
        feature_name = FEATURE_MAP.get(sensor_type, sensor_type)
        if feature_name in features:
            feature_vector[feature_name] = float(value)

        df = pd.DataFrame([feature_vector])
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        anomaly_flag = int(prediction == -1)
        status = "Anomaly" if anomaly_flag else "Normal"

        record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sensor_type': feature_name,
            'sensor_id': sensor_id,
            'value': float(value),
            'anomaly': anomaly_flag
        }

        pd.DataFrame([record]).to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

        return jsonify({
            "sensor_id": sensor_id,
            "sensor_type": feature_name,
            "value": value,
            "prediction": status
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/dashboard-data')
def dashboard_data():
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify([])
        df = pd.read_csv(CSV_FILE)
        if 'anomaly' not in df.columns:
            df['anomaly'] = 0

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp', ascending=True).tail(100)

        result = []
        for _, row in df.iterrows():
            result.append({
                'sensor_id': row.get('sensor_id', ''),
                'sensor_type': row.get('sensor_type', ''),
                'value': float(row.get('value', 0)),
                'timestamp': row['timestamp'].strftime("%Y-%m-%dT%H:%M:%S") if not pd.isnull(row['timestamp']) else '',
                'anomaly': int(row.get('anomaly', 0))
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download_file():
    if not os.path.exists(CSV_FILE):
        return "No data file found", 404
    return send_file(CSV_FILE, as_attachment=True)

@app.route('/static/anomaly.mp3')
def anomaly_mp3():
    path = os.path.join('static', 'anomaly.mp3')
    range_header = request.headers.get('Range', None)
    if not os.path.exists(path):
        abort(404)
    if not range_header:
        return send_file(path)
    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    m = None
    import re
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if m:
        g = m.groups()
        byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])
    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1
    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)
    rv = Response(data, 206, mimetype='audio/mpeg', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

if __name__ == "__main__":
    app.run(debug=True)
