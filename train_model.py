import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Path to the CSV file (same folder)
CSV_PATH = 'sensor_data.csv'

# Load data
df = pd.read_csv(CSV_PATH)

# Ensure required columns exist
required_columns = {'timestamp', 'sensor_type', 'value'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV is missing required columns: {required_columns - set(df.columns)}")

# Filter only numeric 'value'
df = df[pd.to_numeric(df['value'], errors='coerce').notnull()]
df['value'] = df['value'].astype(float)

# Pivot data: one column per sensor type
pivot_df = df.pivot_table(index='timestamp', columns='sensor_type', values='value', aggfunc='mean')
pivot_df = pivot_df.fillna(0)

# Check if pivot_df is empty
if pivot_df.empty:
    raise ValueError("No valid numeric data available after pivot. Check your input CSV.")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_df)

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# Save model and scaler to 'models/' folder
joblib.dump(model, 'model/isolation_forest_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved to 'models/'!")
