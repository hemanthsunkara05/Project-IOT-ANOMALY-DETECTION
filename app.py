from flask import Flask, request, jsonify, render_template, send_file
import csv
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = 'sensor_data.csv'

# POST route to receive sensor data
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Add timestamp if not provided
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add default sensor_id if not given (optional)
    sensor_id = data.get('sensor_id', 'default_sensor')
    value = data.get('value', '')

    # Append to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([sensor_id, value, data['timestamp']])

    return jsonify({'message': 'Data received and saved'}), 200

# GET route to return data as JSON
@app.route('/data', methods=['GET'])
def serve_data():
    result = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 3:
                    result.append({
                        'sensor_id': row[0],
                        'value': row[1],
                        'timestamp': row[2]
                    })
    return jsonify(result)

# Route to render dashboard HTML
@app.route('/dashboard')
def dashboard():
    timestamps = []
    values = []
    try:
        with open('sensor_data.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip header row: ['timestamp', 'sensor_type', 'value']
            for row in reader:
                if len(row) >= 3:
                    timestamps.append(row[0])  # timestamp
                    try:
                        values.append(float(row[2]))  # value
                    except ValueError:
                        values.append(0.0)  # or skip/break/log as needed
    except FileNotFoundError:
        timestamps, values = [], []

    return render_template('dashboard.html', timestamps=timestamps, values=values)


# Route to download the CSV file
@app.route('/download')
def download_file():
    return send_file(CSV_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
