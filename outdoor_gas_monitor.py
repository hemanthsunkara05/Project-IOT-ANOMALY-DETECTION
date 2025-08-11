import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class OutdoorGasMonitor:
    def __init__(self, config_file='gas_monitor_config.json'):
        """Initialize the outdoor gas monitoring system"""
        self.config = self.load_config(config_file)
        self.gas_type = self.config['default_gas_type']
        self.thresholds = self.config['gas_types'][self.gas_type]
        
        # Signal processing buffers
        self.moving_avg_window = self.config['signal_processing']['moving_average_window']
        self.rolling_median_window = self.config['signal_processing']['rolling_median_window']
        self.rate_of_rise_window = self.config['signal_processing']['rate_of_rise_window']
        self.rate_of_rise_threshold = self.config['signal_processing']['rate_of_rise_threshold']
        
        # Data storage
        self.raw_readings = deque(maxlen=max(self.moving_avg_window, self.rolling_median_window) * 2)
        self.timestamps = deque(maxlen=max(self.moving_avg_window, self.rolling_median_window) * 2)
        
        # State tracking
        self.current_state = 'NORMAL'  # NORMAL, EARLY_WARNING, DANGER
        self.consecutive_clear_samples = 0
        self.sustained_high_samples = 0
        self.last_alert_time = None
        
        # Hysteresis parameters
        self.hysteresis_percentage = self.config['hysteresis']['clear_threshold_percentage']
        self.clear_samples_required = self.config['hysteresis']['consecutive_samples_required']
        
        # Voting parameters
        self.min_indicators = self.config['voting']['min_indicators_required']
        self.sustained_samples_required = self.config['voting']['sustained_high_samples']
        
        print(f"✅ Outdoor Gas Monitor initialized for {self.thresholds['name']}")
        print(f"   Low threshold: {self.thresholds['low_threshold']} ppm")
        print(f"   High threshold: {self.thresholds['high_threshold']} ppm")
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_file} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "gas_types": {
                "lpg": {"name": "LPG", "low_threshold": 1000, "high_threshold": 3000},
                "methane": {"name": "Methane", "low_threshold": 2500, "high_threshold": 7500}
            },
            "default_gas_type": "methane",
            "signal_processing": {
                "moving_average_window": 10,
                "rolling_median_window": 300,
                "rate_of_rise_threshold": 500,
                "rate_of_rise_window": 10
            },
            "hysteresis": {"clear_threshold_percentage": 10, "consecutive_samples_required": 3},
            "voting": {"min_indicators_required": 2, "sustained_high_samples": 3},
            "alerts": {
                "early_warning_message": "Early Warning: {gas_type} levels elevated - {value} ppm",
                "danger_message": "DANGER: {gas_type} levels critical - {value} ppm - EVACUATE IMMEDIATELY!",
                "rate_of_rise_message": "Rapid {gas_type} increase detected - {value} ppm (+{increase} ppm in {time}s)"
            },
            "logging": {
                "log_raw_values": True,
                "log_processed_values": True,
                "log_alerts": True,
                "csv_filename": "outdoor_gas_monitor.csv"
            }
        }
    
    def set_gas_type(self, gas_type):
        """Change the gas type being monitored"""
        if gas_type in self.config['gas_types']:
            self.gas_type = gas_type
            self.thresholds = self.config['gas_types'][gas_type]
            print(f"✅ Switched to monitoring {self.thresholds['name']}")
            return True
        else:
            print(f"❌ Unknown gas type: {gas_type}")
            return False
    
    def add_reading(self, ppm_value, timestamp=None):
        """Add a new gas reading and process it"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store raw reading
        self.raw_readings.append(ppm_value)
        self.timestamps.append(timestamp)
        
        # Process the reading
        result = self.process_reading(ppm_value, timestamp)
        
        # Log if enabled
        if self.config['logging']['log_raw_values']:
            self.log_reading(ppm_value, timestamp, result)
        
        return result
    
    def process_reading(self, ppm_value, timestamp):
        """Process a gas reading and determine alert status"""
        if len(self.raw_readings) < 2:
            return self.create_result('INSUFFICIENT_DATA', ppm_value, timestamp)
        
        # Calculate signal processing values
        moving_avg = self.calculate_moving_average()
        rolling_median = self.calculate_rolling_median()
        rate_of_rise = self.calculate_rate_of_rise()
        
        # Determine indicators
        indicators = self.evaluate_indicators(ppm_value, moving_avg, rate_of_rise)
        
        # Apply voting logic
        alert_decision = self.apply_voting_logic(indicators, moving_avg)
        
        # Apply hysteresis
        final_state = self.apply_hysteresis(alert_decision, moving_avg)
        
        # Create result
        result = self.create_result(final_state, ppm_value, timestamp, {
            'moving_average': moving_avg,
            'rolling_median': rolling_median,
            'rate_of_rise': rate_of_rise,
            'indicators': indicators,
            'voting_decision': alert_decision
        })
        
        # Update state
        self.update_state(final_state, moving_avg)
        
        return result
    
    def calculate_moving_average(self):
        """Calculate moving average over the specified window"""
        if len(self.raw_readings) < self.moving_avg_window:
            return np.mean(list(self.raw_readings))
        
        recent_readings = list(self.raw_readings)[-self.moving_avg_window:]
        return np.mean(recent_readings)
    
    def calculate_rolling_median(self):
        """Calculate rolling median over the specified window"""
        if len(self.raw_readings) < self.rolling_median_window:
            return np.median(list(self.raw_readings))
        
        recent_readings = list(self.raw_readings)[-self.rolling_median_window:]
        return np.median(recent_readings)
    
    def calculate_rate_of_rise(self):
        """Calculate rate of rise over the specified window"""
        if len(self.raw_readings) < 2:
            return 0
        
        if len(self.raw_readings) < self.rate_of_rise_window:
            # Use all available data
            readings = list(self.raw_readings)
            timestamps = list(self.timestamps)
        else:
            # Use only the rate of rise window
            readings = list(self.raw_readings)[-self.rate_of_rise_window:]
            timestamps = list(self.timestamps)[-self.rate_of_rise_window:]
        
        if len(readings) < 2:
            return 0
        
        # Calculate time difference in seconds
        time_diff = (timestamps[-1] - timestamps[0]).total_seconds()
        if time_diff == 0:
            return 0
        
        # Calculate ppm difference
        ppm_diff = readings[-1] - readings[0]
        
        return ppm_diff / time_diff  # ppm per second
    
    def evaluate_indicators(self, current_value, moving_avg, rate_of_rise):
        """Evaluate all indicators for voting logic"""
        indicators = {
            'threshold_met': False,
            'rate_of_rise_triggered': False,
            'sustained_high': False
        }
        
        # Indicator 1: Gas level threshold met
        if moving_avg >= self.thresholds['low_threshold']:
            indicators['threshold_met'] = True
        
        # Indicator 2: Rate of rise triggered
        if (rate_of_rise > self.rate_of_rise_threshold and 
            current_value >= self.thresholds['low_threshold']):
            indicators['rate_of_rise_triggered'] = True
        
        # Indicator 3: Sustained high reading
        if moving_avg >= self.thresholds['low_threshold']:
            self.sustained_high_samples += 1
        else:
            self.sustained_high_samples = 0
        
        if self.sustained_high_samples >= self.sustained_samples_required:
            indicators['sustained_high'] = True
        
        return indicators
    
    def apply_voting_logic(self, indicators, moving_avg):
        """Apply voting logic to determine alert decision"""
        active_indicators = sum(indicators.values())
        
        if active_indicators < self.min_indicators:
            return 'NORMAL'
        
        # Determine alert level based on moving average
        if moving_avg >= self.thresholds['high_threshold']:
            return 'DANGER'
        elif moving_avg >= self.thresholds['low_threshold']:
            return 'EARLY_WARNING'
        else:
            return 'NORMAL'
    
    def apply_hysteresis(self, alert_decision, moving_avg):
        """Apply hysteresis to prevent alert flickering"""
        if alert_decision == 'NORMAL':
            # Check if we should clear the alarm
            if self.current_state in ['EARLY_WARNING', 'DANGER']:
                clear_threshold = self.thresholds['low_threshold'] * (1 - self.hysteresis_percentage / 100)
                
                if moving_avg < clear_threshold:
                    self.consecutive_clear_samples += 1
                else:
                    self.consecutive_clear_samples = 0
                
                if self.consecutive_clear_samples >= self.clear_samples_required:
                    return 'NORMAL'
                else:
                    return self.current_state  # Keep current alarm state
            else:
                return 'NORMAL'
        else:
            # New alarm condition
            self.consecutive_clear_samples = 0
            return alert_decision
    
    def update_state(self, new_state, moving_avg):
        """Update the current state and trigger alerts if needed"""
        if new_state != self.current_state:
            self.current_state = new_state
            self.last_alert_time = datetime.now()
            
            # Generate alert message
            if new_state == 'EARLY_WARNING':
                message = self.config['alerts']['early_warning_message'].format(
                    gas_type=self.thresholds['name'],
                    value=f"{moving_avg:.1f}"
                )
            elif new_state == 'DANGER':
                message = self.config['alerts']['danger_message'].format(
                    gas_type=self.thresholds['name'],
                    value=f"{moving_avg:.1f}"
                )
            else:
                message = f"Gas levels normal: {moving_avg:.1f} ppm"
            
            print(f"🚨 {message}")
    
    def create_result(self, state, ppm_value, timestamp, additional_data=None):
        """Create a standardized result dictionary"""
        result = {
            'timestamp': timestamp,
            'raw_value': ppm_value,
            'state': state,
            'gas_type': self.thresholds['name'],
            'alert_triggered': state in ['EARLY_WARNING', 'DANGER'],
            'thresholds': {
                'low': self.thresholds['low_threshold'],
                'high': self.thresholds['high_threshold']
            }
        }
        
        if additional_data:
            result.update(additional_data)
        
        return result
    
    def log_reading(self, ppm_value, timestamp, result):
        """Log reading to CSV file"""
        try:
            log_data = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'raw_ppm': ppm_value,
                'moving_average': result.get('moving_average', 0),
                'rolling_median': result.get('rolling_median', 0),
                'rate_of_rise': result.get('rate_of_rise', 0),
                'state': result['state'],
                'gas_type': result['gas_type']
            }
            
            df = pd.DataFrame([log_data])
            filename = self.config['logging']['csv_filename']
            
            # Append to CSV file
            df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
            
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def get_status(self):
        """Get current system status"""
        return {
            'gas_type': self.thresholds['name'],
            'current_state': self.current_state,
            'thresholds': self.thresholds,
            'recent_readings': len(self.raw_readings),
            'last_alert_time': self.last_alert_time
        }
    
    def test_system(self):
        """Test the system with various scenarios"""
        print("🧪 Testing Outdoor Gas Monitor System")
        print("=" * 50)
        
        test_scenarios = [
            (500, "Normal baseline"),
            (1500, "Below low threshold"),
            (2500, "At low threshold (Early Warning)"),
            (5000, "Between thresholds"),
            (8000, "Above high threshold (Danger)"),
            (12000, "Very high reading")
        ]
        
        for ppm, description in test_scenarios:
            print(f"\n📊 Testing: {ppm} ppm ({description})")
            result = self.add_reading(ppm)
            
            status = "🚨 ALERT" if result['alert_triggered'] else "✅ NORMAL"
            print(f"   Status: {status}")
            print(f"   State: {result['state']}")
            print(f"   Moving Average: {result.get('moving_average', 0):.1f} ppm")
            print(f"   Rate of Rise: {result.get('rate_of_rise', 0):.1f} ppm/s")
            
            if result['alert_triggered']:
                print(f"   🚨 {result['gas_type']} Alert: {result['state']}")

if __name__ == "__main__":
    monitor = OutdoorGasMonitor()
    monitor.test_system()
