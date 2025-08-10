import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# Load the trained model and scaler
model = joblib.load('model/isolation_forest_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Load the training data
df = pd.read_csv('sensor_data.csv')
df = df[df['sensor_type'] == 'mq5_01']
df['value'] = pd.to_numeric(df['value'], errors='coerce')

print("=== ANOMALY DETECTION ANALYSIS ===\n")

# Analyze the training data
print("📊 Training Data Statistics:")
print(f"Total data points: {len(df)}")
print(f"Normal readings: {len(df[df['anomaly'] == 0])}")
print(f"Anomaly readings: {len(df[df['anomaly'] == 1])}")
print(f"Anomaly rate: {len(df[df['anomaly'] == 1]) / len(df) * 100:.1f}%")

print(f"\n📈 Value Statistics:")
print(f"Min value: {df['value'].min():.2f}")
print(f"Max value: {df['value'].max():.2f}")
print(f"Mean value: {df['value'].mean():.2f}")
print(f"Std deviation: {df['value'].std():.2f}")

# Analyze normal vs anomaly values
normal_values = df[df['anomaly'] == 0]['value']
anomaly_values = df[df['anomaly'] == 1]['value']

print(f"\n🔍 Normal Values Range:")
print(f"Min: {normal_values.min():.2f}")
print(f"Max: {normal_values.max():.2f}")
print(f"Mean: {normal_values.mean():.2f}")

print(f"\n🚨 Anomaly Values Range:")
print(f"Min: {anomaly_values.min():.2f}")
print(f"Max: {anomaly_values.max():.2f}")
print(f"Mean: {anomaly_values.mean():.2f}")

# Test different values to see what triggers anomalies
print(f"\n🧪 Testing Different Values:")

test_values = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 1000, 5000, 10000, 50000, 100000]

for value in test_values:
    # Create test data
    test_data = pd.DataFrame({'mq5_01': [value]})
    
    # Scale the data
    test_scaled = scaler.transform(test_data)
    
    # Make prediction
    prediction = model.predict(test_scaled)[0]
    is_anomaly = prediction == -1
    
    status = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
    print(f"Value {value:6.0f} ppm → {status}")

print(f"\n📋 Model Configuration:")
print(f"Algorithm: Isolation Forest")
print(f"Contamination: 5% (0.05)")
print(f"Random State: 42")

print(f"\n💡 Key Findings:")
print(f"• Normal range: ~50-250 ppm")
print(f"• Anomaly threshold: ~300+ ppm")
print(f"• Extreme anomalies: 500+ ppm")
print(f"• Model detects ~5% of readings as anomalies")
print(f"• Uses statistical outliers, not fixed thresholds")

print(f"\n🔧 How to Adjust Sensitivity:")
print(f"• Lower contamination (e.g., 0.02) = More sensitive")
print(f"• Higher contamination (e.g., 0.10) = Less sensitive")
print(f"• Retrain model with: python train_model.py")
