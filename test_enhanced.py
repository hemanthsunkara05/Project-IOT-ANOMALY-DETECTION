#!/usr/bin/env python3
"""
Simple test script for the enhanced anomaly detector
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_anomaly_detector import EnhancedAnomalyDetector
    print("✅ Enhanced detector imported successfully")
    
    # Test basic functionality
    detector = EnhancedAnomalyDetector()
    print("✅ Enhanced detector initialized")
    
    # Test with a simple value
    result = detector.comprehensive_anomaly_detection(500)
    print(f"✅ Test completed: {result['anomaly_type']} (confidence: {result['confidence']:.1%})")
    
    print("\n🎉 Enhanced anomaly detector is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
