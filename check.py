import joblib

# Load the model and scaler using joblib
model = joblib.load("model/isolation_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")

print("✅ Model and scaler loaded successfully.")
