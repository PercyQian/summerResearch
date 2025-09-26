#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LMP预测应用的基本功能
"""

import os
import sys

def test_imports():
    """测试必要的模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit 导入成功")
    except ImportError:
        print("❌ Streamlit 导入失败，请安装: pip install streamlit")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly 导入成功")
    except ImportError:
        print("❌ Plotly 导入失败，请安装: pip install plotly")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ 基础数据包导入成功")
    except ImportError:
        print("❌ 基础数据包导入失败")
        return False
    
    return True

def test_model_files():
    """测试模型文件是否存在"""
    print("\n🧪 测试模型文件...")
    
    model_dir = "saved_models"
    required_files = [
        'xgboost_model.pkl',
        'scaler.pkl', 
        'feature_columns.pkl',
        'k_value.pkl',
        'model_info.pkl'
    ]
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录 {model_dir} 不存在")
        return False
    
    missing_files = []
    for file in required_files:
        if os.path.exists(os.path.join(model_dir, file)):
            print(f"✅ {file} 存在")
        else:
            missing_files.append(file)
            print(f"❌ {file} 缺失")
    
    return len(missing_files) == 0

def test_main_module():
    """测试主训练模块"""
    print("\n🧪 测试主训练模块...")
    
    try:
        from power_market_anomaly_detection import load_model_for_prediction, apply_holiday_features
        print("✅ 主模块函数导入成功")
        return True
    except Exception as e:
        print(f"❌ 主模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试LMP预测应用...")
    
    # 测试导入
    if not test_imports():
        print("\n❌ 模块导入测试失败")
        return
    
    # 测试主模块
    if not test_main_module():
        print("\n❌ 主模块测试失败")
        return
    
    # 测试模型文件
    if not test_model_files():
        print("\n❌ 模型文件测试失败")
        print("💡 请先运行: python power_market_anomaly_detection.py")
        return
    
    print("\n✅ 所有测试通过！")
    print("\n🚀 您可以运行以下命令启动应用:")
    print("python run_prediction_app.py")
    print("或者:")
    print("streamlit run lmp_prediction_app.py")

if __name__ == "__main__":
    main() 