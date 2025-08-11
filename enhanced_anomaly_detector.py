import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnomalyDetector:
    def __init__(self, csv_file='sensor_data.csv'):
        self.csv_file = csv_file
        self.model = None
        self.scaler = None
        self.load_models()
        
        # Thresholds for different detection methods
        self.thresholds = {
            'absolute_warning': 300,    # ppm - Warning level
            'absolute_critical': 500,   # ppm - Critical level
            'absolute_extreme': 1000,   # ppm - Extreme level
            'trend_window': 5,          # Number of readings to check for trend
            'trend_threshold': 0.1,     # 10% increase threshold
            'consecutive_increases': 3,  # Number of consecutive increases to trigger
            
            # Home Safety Thresholds for MQ-5 sensor
            'home_safety_low': 2000,    # ppm - Early warning for LPG/methane/natural gas leak
            'home_safety_high': 4000    # ppm - Danger level - evacuate immediately
        }
    
    def load_models(self):
        """Load the trained Isolation Forest model and scaler"""
        try:
            self.model = joblib.load('model/isolation_forest_model.pkl')
            self.scaler = joblib.load('model/scaler.pkl')
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load models: {e}")
            self.model = None
            self.scaler = None
    
    def detect_statistical_anomaly(self, value, sensor_type='mq5_01'):
        """Detect anomalies using Isolation Forest (original method)"""
        if self.model is None or self.scaler is None:
            return False
        
        try:
            # Create feature vector
            features = {'mq5_01': 0.0}
            features[sensor_type] = float(value)
            
            df = pd.DataFrame([features])
            scaled_data = self.scaler.transform(df)
            prediction = self.model.predict(scaled_data)[0]
            
            return prediction == -1  # -1 indicates anomaly
        except Exception as e:
            print(f"Statistical anomaly detection error: {e}")
            return False
    
    def detect_absolute_threshold_anomaly(self, value):
        """Detect anomalies based on absolute thresholds"""
        value = float(value)
        
        if value >= self.thresholds['absolute_extreme']:
            return True, 'EXTREME'
        elif value >= self.thresholds['absolute_critical']:
            return True, 'CRITICAL'
        elif value >= self.thresholds['absolute_warning']:
            return True, 'WARNING'
        else:
            return False, 'NORMAL'
    
    def detect_home_safety_anomaly(self, value):
        """Detect home safety anomalies based on MQ-5 sensor thresholds"""
        value = float(value)
        
        if value >= self.thresholds['home_safety_high']:
            return True, 'DANGER_EVACUATE', f'Danger level reached: {value} ppm - EVACUATE IMMEDIATELY!'
        elif value >= self.thresholds['home_safety_low']:
            return True, 'EARLY_WARNING', f'Early warning: {value} ppm - Potential gas leak detected'
        else:
            return False, 'SAFE', f'Safe level: {value} ppm'
    
    def detect_trend_anomaly(self, sensor_type='mq5_01', window=None):
        """Detect anomalies based on increasing trends"""
        if window is None:
            window = self.thresholds['trend_window']
        
        try:
            # Load recent data
            df = pd.read_csv(self.csv_file)
            sensor_data = df[df['sensor_type'] == sensor_type].tail(window + 1)
            
            if len(sensor_data) < window:
                return False, 'INSUFFICIENT_DATA'
            
            values = sensor_data['value'].astype(float).values
            
            # Check for consecutive increases
            consecutive_increases = 0
            for i in range(1, len(values)):
                if values[i] > values[i-1]:
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0
            
            # Check for overall trend
            if len(values) >= 3:
                recent_values = values[-3:]
                trend_increase = (recent_values[-1] - recent_values[0]) / recent_values[0]
                
                # Trigger if too many consecutive increases or significant trend
                if consecutive_increases >= self.thresholds['consecutive_increases']:
                    return True, f'TREND_CONSECUTIVE_{consecutive_increases}'
                elif trend_increase >= self.thresholds['trend_threshold']:
                    return True, f'TREND_INCREASE_{trend_increase:.1%}'
            
            return False, 'NO_TREND'
            
        except Exception as e:
            print(f"Trend anomaly detection error: {e}")
            return False, 'ERROR'
    
    def detect_velocity_anomaly(self, sensor_type='mq5_01', window=3):
        """Detect anomalies based on rate of change (velocity)"""
        try:
            df = pd.read_csv(self.csv_file)
            sensor_data = df[df['sensor_type'] == sensor_type].tail(window + 1)
            
            if len(sensor_data) < 2:
                return False, 'INSUFFICIENT_DATA'
            
            # Calculate rate of change
            values = sensor_data['value'].astype(float).values
            timestamps = pd.to_datetime(sensor_data['timestamp'])
            
            # Calculate ppm per minute
            time_diff = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 60
            value_diff = values[-1] - values[0]
            velocity = value_diff / time_diff if time_diff > 0 else 0
            
            # High velocity threshold (e.g., 50 ppm per minute)
            if velocity > 50:
                return True, f'HIGH_VELOCITY_{velocity:.1f}_ppm_min'
            
            return False, f'VELOCITY_{velocity:.1f}_ppm_min'
            
        except Exception as e:
            print(f"Velocity anomaly detection error: {e}")
            return False, 'ERROR'
    
    def comprehensive_anomaly_detection(self, value, sensor_type='mq5_01'):
        """Comprehensive anomaly detection using all methods"""
        results = {
            'value': value,
            'sensor_type': sensor_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'anomaly_detected': False,
            'anomaly_type': 'NORMAL',
            'confidence': 0,
            'details': {}
        }
        
        # Method 1: Absolute Threshold Detection
        absolute_anomaly, absolute_type = self.detect_absolute_threshold_anomaly(value)
        results['details']['absolute'] = {
            'detected': absolute_anomaly,
            'type': absolute_type
        }
        
        # Method 2: Statistical Anomaly Detection
        statistical_anomaly = self.detect_statistical_anomaly(value, sensor_type)
        results['details']['statistical'] = {
            'detected': bool(statistical_anomaly),
            'type': 'ISOLATION_FOREST'
        }
        
        # Method 3: Trend Anomaly Detection
        trend_anomaly, trend_type = self.detect_trend_anomaly(sensor_type)
        results['details']['trend'] = {
            'detected': trend_anomaly,
            'type': trend_type
        }
        
        # Method 4: Velocity Anomaly Detection
        velocity_anomaly, velocity_type = self.detect_velocity_anomaly(sensor_type)
        results['details']['velocity'] = {
            'detected': velocity_anomaly,
            'type': velocity_type
        }
        
        # Method 5: Home Safety Detection (for MQ-5 sensors)
        home_safety_anomaly, home_safety_type, home_safety_message = self.detect_home_safety_anomaly(value)
        results['details']['home_safety'] = {
            'detected': home_safety_anomaly,
            'type': home_safety_type,
            'message': home_safety_message
        }
        
        # Determine overall anomaly status
        anomaly_count = sum([
            absolute_anomaly,
            bool(statistical_anomaly),
            trend_anomaly,
            velocity_anomaly,
            home_safety_anomaly
        ])
        
        # Calculate confidence based on number of methods that detected anomalies
        results['confidence'] = anomaly_count / 5.0
        
        # Determine anomaly type based on priority (Home Safety has highest priority)
        if home_safety_anomaly and 'DANGER_EVACUATE' in home_safety_type:
            results['anomaly_type'] = 'DANGER_EVACUATE'
            results['anomaly_detected'] = True
            results['alert_message'] = home_safety_message
        elif home_safety_anomaly and 'EARLY_WARNING' in home_safety_type:
            results['anomaly_type'] = 'EARLY_WARNING'
            results['anomaly_detected'] = True
            results['alert_message'] = home_safety_message
        elif absolute_anomaly and 'EXTREME' in absolute_type:
            results['anomaly_type'] = 'EXTREME'
            results['anomaly_detected'] = True
        elif absolute_anomaly and 'CRITICAL' in absolute_type:
            results['anomaly_type'] = 'CRITICAL'
            results['anomaly_detected'] = True
        elif trend_anomaly and 'TREND' in trend_type:
            results['anomaly_type'] = 'TREND'
            results['anomaly_detected'] = True
        elif velocity_anomaly and 'HIGH_VELOCITY' in velocity_type:
            results['anomaly_type'] = 'HIGH_VELOCITY'
            results['anomaly_detected'] = True
        elif absolute_anomaly and 'WARNING' in absolute_type:
            results['anomaly_type'] = 'WARNING'
            results['anomaly_detected'] = True
        elif bool(statistical_anomaly):
            results['anomaly_type'] = 'STATISTICAL'
            results['anomaly_detected'] = True
        elif results['confidence'] >= 0.5:  # Multiple methods detected something
            results['anomaly_type'] = 'MULTIPLE_INDICATORS'
            results['anomaly_detected'] = True
        
        return results
    
    def test_detection_system(self):
        """Test the detection system with various scenarios"""
        print("🧪 Testing Enhanced Anomaly Detection System")
        print("=" * 50)
        
        test_cases = [
            (50, "Normal baseline"),
            (250, "Normal upper range"),
            (300, "Warning threshold"),
            (500, "Critical threshold"),
            (1000, "Extreme threshold"),
            (1500, "Very high value"),
            (2000, "Home safety low threshold (Early Warning)"),
            (3000, "Between home safety thresholds"),
            (4000, "Home safety high threshold (Danger - Evacuate)"),
            (5000, "Above home safety high threshold")
        ]
        
        for value, description in test_cases:
            print(f"\n📊 Testing: {value} ppm ({description})")
            result = self.comprehensive_anomaly_detection(value)
            
            status = "🚨 ANOMALY" if result['anomaly_detected'] else "✅ NORMAL"
            print(f"   Status: {status}")
            print(f"   Type: {result['anomaly_type']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            if result['anomaly_detected']:
                print("   Details:")
                for method, details in result['details'].items():
                    if details['detected']:
                        if method == 'home_safety' and 'message' in details:
                            print(f"     - {method.upper()}: {details['type']} - {details['message']}")
                        else:
                            print(f"     - {method.upper()}: {details['type']}")
                
                # Show alert message if available
                if 'alert_message' in result:
                    print(f"   🚨 ALERT: {result['alert_message']}")

if __name__ == "__main__":
    detector = EnhancedAnomalyDetector()
    detector.test_detection_system()
