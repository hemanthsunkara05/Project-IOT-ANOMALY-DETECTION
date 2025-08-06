import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

CSV_PATH = 'sensor_data.csv'

# Load dataset
df = pd.read_csv(CSV_PATH)

# Verify required columns
required_columns = {'timestamp', 'sensor_type', 'value'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV is missing required columns: {required_columns - set(df.columns)}")

# Filter to include only MQ-5 sensor data
df = df[df['sensor_type'] == 'mq5_01']

# Ensure 'value' column is numeric
df = df[pd.to_numeric(df['value'], errors='coerce').notnull()]
df['value'] = df['value'].astype(float)

# Pivot the data: for one sensor type, set index and rename
pivot_df = df.set_index('timestamp')[['value']]
pivot_df.columns = ['mq5_01']
pivot_df = pivot_df.fillna(0)

if pivot_df.empty:
    raise ValueError("No valid MQ-5 numeric data available for training.")

print(f"Training only on these features: {list(pivot_df.columns)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_df)

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# Ensure 'model' directory exists
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model and scaler
joblib.dump(model, 'model/isolation_forest_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved successfully in 'model/' folder!")
