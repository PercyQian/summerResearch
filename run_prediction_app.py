#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin to run the prediction app"""

import os
import sys
import subprocess

def check_model_files():
    """check if model files exist"""
    model_dir = "saved_models"
    required_files = [
        'xgboost_model.pkl',
        'scaler.pkl', 
        'feature_columns.pkl',
        'k_value.pkl',
        'model_info.pkl'
    ]
    
    if not os.path.exists(model_dir):
        print(f"❌ model directory {model_dir} does not exist")
        print("please run the main training script to train and save models")
        return False
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ missing the following model files: {', '.join(missing_files)}")
        print("please run the main training script to train and save models")
        return False
    
    return True

def main():
    """main function"""
    print("🚀 start LMP anomaly detection prediction app...")
    
    # check model files
    if not check_model_files():
        print("\n💡 hint: run the following command to train models:")
        print("python power_market_anomaly_detection.py")
        return
    
    print("✅ model files check passed, start Streamlit app...")
    
    # start Streamlit app
    try:
        # use subprocess to run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "lmp_prediction_app.py", "--server.port=8501"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 app stopped")
    except Exception as e:
        print(f"❌ start app failed: {e}")
        print("\n💡 please ensure Streamlit is installed:")
        print("pip install streamlit plotly")
        print("pip install xgboost")
        print("pip install scikit-learn")
        print("pip install pandas")
        print("pip install numpy")
        print("pip install matplotlib")
        print("pip install seaborn")

if __name__ == "__main__":
    main() 