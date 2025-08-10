from flask import Flask, redirect, request, jsonify, render_template, send_file, Response, abort, send_from_directory
import os
import pandas as pd
import joblib
from datetime import datetime
import pickle
from enhanced_anomaly_detector import EnhancedAnomalyDetector

app = Flask(__name__)

FEATURE_MAP = {
    "MQ-5": "mq5_01",
    "Gas": "mq5_01",
    "Temperature": "temp_01",
    "Humidity": "humidity_01",
    "Pressure": "pressure_01",
    "Light": "light_01",
    "Motion": "motion_01"
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

# Initialize enhanced anomaly detector
try:
    enhanced_detector = EnhancedAnomalyDetector(CSV_FILE)
    print("✅ Enhanced anomaly detector initialized")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize enhanced detector: {e}")
    enhanced_detector = None

@app.route('/')
def home():
    return redirect("/modern")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/modern')
def modern_dashboard():
    return render_template("modern_dashboard.html")

# New route to serve the React frontend
@app.route('/react')
def react_dashboard():
    return render_template("react_dashboard.html")

# Serve static files from the React frontend
@app.route('/frontend/<path:filename>')
def react_static(filename):
    return send_from_directory('frontend', filename)

# Serve React build files (when available)
@app.route('/static/<path:filename>')
def react_build_static(filename):
    return send_from_directory('frontend/dist', filename)

# API endpoint for sensor data (compatible with React frontend)
@app.route('/api/sensors', methods=['GET'])
def get_sensors():
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify([])
        
        df = pd.read_csv(CSV_FILE)
        if 'anomaly' not in df.columns:
            df['anomaly'] = 0

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp', ascending=True).tail(100)

        # Group by sensor type and get latest readings
        sensors = []
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type].iloc[-1]
            
            # Map sensor type to display name
            display_name = next((k for k, v in FEATURE_MAP.items() if v == sensor_type), sensor_type)
            
            sensors.append({
                'id': f'sensor-{sensor_type}',
                'type': display_name,
                'value': float(sensor_data.get('value', 0)),
                'unit': get_unit_for_sensor(display_name),
                'timestamp': sensor_data['timestamp'].strftime("%Y-%m-%dT%H:%M:%S") if not pd.isnull(sensor_data['timestamp']) else '',
                'isAnomaly': int(sensor_data.get('anomaly', 0)),
                'trend': 'stable',  # You can implement trend calculation
                'icon': get_icon_for_sensor(display_name)
            })
        
        return jsonify(sensors)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for historical data
@app.route('/api/sensors/<sensor_id>/history', methods=['GET'])
def get_sensor_history(sensor_id):
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify([])
        
        df = pd.read_csv(CSV_FILE)
        if 'anomaly' not in df.columns:
            df['anomaly'] = 0

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Extract sensor type from sensor_id
        sensor_type = sensor_id.replace('sensor-', '')
        
        # Filter data for this sensor type
        sensor_data = df[df['sensor_type'] == sensor_type].tail(24)  # Last 24 readings
        
        history = []
        for _, row in sensor_data.iterrows():
            history.append({
                'timestamp': row['timestamp'].strftime("%H:%M") if not pd.isnull(row['timestamp']) else '',
                'value': float(row.get('value', 0)),
                'isAnomaly': int(row.get('anomaly', 0))
            })
        
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper functions
def get_unit_for_sensor(sensor_type):
    units = {
        'Gas': 'ppm',
        'Temperature': '°C',
        'Humidity': '%',
        'Pressure': 'kPa',
        'Light': 'lux',
        'Motion': 'events/min'
    }
    return units.get(sensor_type, '')

def get_icon_for_sensor(sensor_type):
    icons = {
        'Gas': '🌬️',
        'Temperature': '🌡️',
        'Humidity': '💧',
        'Pressure': '📊',
        'Light': '💡',
        'Motion': '🏃'
    }
    return icons.get(sensor_type, '📡')

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

    # Use enhanced anomaly detection if available
    if enhanced_detector is not None:
        try:
            result = enhanced_detector.comprehensive_anomaly_detection(value, sensor_type)
            prediction = int(result['anomaly_detected'])
            
            # Add detailed anomaly information to the response
            response_data = {
                'anomaly': prediction,
                'anomaly_type': result['anomaly_type'],
                'confidence': result['confidence'],
                'details': result['details'],
                'message': f'Enhanced detection: {result["anomaly_type"]} (confidence: {result["confidence"]:.1%})'
            }
        except Exception as e:
            print(f"Enhanced detection failed: {e}")
            # Fallback to original method
            prediction = 0
            response_data = {
                'anomaly': prediction,
                'message': 'Enhanced detection failed, using fallback.'
            }
    else:
        # Fallback to original Isolation Forest method
        try:
            pivot_df = all_data.pivot_table(index='timestamp', columns='sensor_type', values='value', aggfunc='mean').fillna(0)
            latest_data = pivot_df.iloc[[-1]]
            latest_scaled = scaler.transform(latest_data)
            prediction = int(model.predict(latest_scaled)[0] == -1)
            response_data = {
                'anomaly': prediction,
                'message': 'Data received and prediction made (original method).'
            }
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    all_data.at[all_data.index[-1], 'anomaly'] = prediction
    all_data.to_csv(CSV_FILE, index=False)

    return jsonify(response_data)

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

@app.route('/add-sample-data')
def add_sample_data():
    """Add sample sensor data for testing the dashboard"""
    import random
    from datetime import datetime, timedelta
    
    # Generate sample data for the last hour
    sample_data = []
    now = datetime.now()
    
    for i in range(30):  # 30 data points
        timestamp = now - timedelta(minutes=i*2)  # Every 2 minutes
        value = random.uniform(20, 80)  # Random value between 20-80
        anomaly = 1 if random.random() < 0.1 else 0  # 10% chance of anomaly
        
        sample_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': 'test_sensor',
            'sensor_type': 'mq5_01',
            'value': round(value, 2),
            'anomaly': anomaly
        })
    
    # Save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
    
    return jsonify({
        'message': f'Added {len(sample_data)} sample data points',
        'data_points': sample_data
    })

@app.route('/test-enhanced-detection')
def test_enhanced_detection():
    """Test the enhanced anomaly detection system"""
    if enhanced_detector is None:
        return jsonify({'error': 'Enhanced detector not available'}), 500
    
    try:
        # Test with various scenarios
        test_results = []
        test_values = [50, 250, 300, 500, 1000, 1500]
        
        for value in test_values:
            result = enhanced_detector.comprehensive_anomaly_detection(value)
            test_results.append({
                'value': value,
                'anomaly_detected': result['anomaly_detected'],
                'anomaly_type': result['anomaly_type'],
                'confidence': result['confidence'],
                'details': result['details']
            })
        
        return jsonify({
            'message': 'Enhanced detection test completed',
            'test_results': test_results
        })
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

@app.route('/add-trend-test-data')
def add_trend_test_data():
    """Add data that creates a gradual increasing trend to test trend detection"""
    import random
    from datetime import datetime, timedelta
    
    # Create a gradual increasing trend
    trend_data = []
    now = datetime.now()
    base_value = 200  # Start at 200 ppm
    
    for i in range(10):  # 10 data points with increasing trend
        timestamp = now - timedelta(minutes=i*2)  # Every 2 minutes
        # Gradually increase from 200 to 600 ppm
        value = base_value + (i * 40) + random.uniform(-5, 5)  # Add some noise
        
        trend_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': 'trend_test_sensor',
            'sensor_type': 'mq5_01',
            'value': round(value, 2),
            'anomaly': 0  # Will be determined by enhanced detector
        })
    
    # Save to CSV
    df = pd.DataFrame(trend_data)
    df.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
    
    return jsonify({
        'message': f'Added {len(trend_data)} trend test data points (gradual increase from {base_value} to {base_value + 360} ppm)',
        'data_points': trend_data
    })

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
