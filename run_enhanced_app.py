#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行增强版预测应用
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查依赖包"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'requests', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("💡 请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 启动增强版电力市场LMP异常检测预测系统")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查模型文件
    if not os.path.exists("saved_models"):
        print("❌ 未找到训练好的模型文件")
        print("💡 请先运行以下命令训练模型:")
        print("python power_market_anomaly_detection.py")
        return
    
    # 启动Streamlit应用
    try:
        print("🌐 启动Web应用...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "enhanced_prediction_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
