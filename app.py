from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

CSV_FILE = 'sensor_data.csv'

# Ensure the CSV file has a header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sensor_id', 'value', 'timestamp'])

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    print("Received:", data)

    # Save to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data.get('sensor_id'), data.get('value'), data.get('timestamp')])

    return jsonify({'message': 'Data saved successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)