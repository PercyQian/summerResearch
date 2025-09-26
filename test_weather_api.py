#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试天气API功能
"""

import requests
import datetime as dt

# 区域坐标
zone_coords = {
    'PECO':  (39.9526,  -75.1652),
}

def test_weather_api():
    """测试所有区域天气API聚合功能"""
    print("🧪 测试所有区域天气API聚合功能...")
    
    # 测试明天的天气预报
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    
    try:
        # 导入应用的天气函数
        from lmp_prediction_app import get_all_zones_weather_forecast
        
        print(f"📡 请求所有13个区域的天气预报 - 日期: {tomorrow.strftime('%Y-%m-%d')}, 时间: 12:00")
        
        weather_data = get_all_zones_weather_forecast(tomorrow, 12)
        
        if weather_data:
            print(f"\n✅ 聚合天气数据获取成功!")
            print(f"🌡️  平均温度: {weather_data['agg_temp_mean']:.1f}°C")
            print(f"🌡️  温度范围: {weather_data['agg_temp_min']:.1f}~{weather_data['agg_temp_max']:.1f}°C")
            print(f"🌡️  温度标准差: {weather_data['agg_temp_std']:.1f}°C")
            print(f"💨 平均风速: {weather_data['agg_wind_mean']:.1f} km/h")
            print(f"💨 最大风速: {weather_data['agg_wind_max']:.1f} km/h")
            print(f"🌊 平均气压: {weather_data['agg_pressure_mean']:.1f} hPa")
            print(f"🌱 平均土壤温度: {weather_data['agg_soil_temp_mean']:.1f}°C")
            print(f"💧 平均土壤湿度: {weather_data['agg_soil_moisture_mean']:.3f} m³/m³")
            print(f"📊 成功获取区域: {weather_data['successful_zones']}/13")
            
            if weather_data['failed_zones']:
                print(f"⚠️  失败区域: {', '.join(weather_data['failed_zones'])}")
            
            return True
        else:
            print(f"❌ 无法获取聚合天气数据")
            return False
            
    except Exception as e:
        print(f"❌ 天气API聚合测试失败: {e}")
        return False

def test_model_k_value():
    """测试保存的模型k值"""
    print("\n🧪 测试模型k值...")
    
    try:
        import pickle
        import os
        
        model_dir = "saved_models"
        k_file = os.path.join(model_dir, "k_value.pkl")
        
        if os.path.exists(k_file):
            with open(k_file, 'rb') as f:
                k_value = pickle.load(f)
            print(f"✅ 模型保存的k值: {k_value}")
            
            # 检查模型信息
            info_file = os.path.join(model_dir, "model_info.pkl")
            if os.path.exists(info_file):
                with open(info_file, 'rb') as f:
                    model_info = pickle.load(f)
                print(f"📊 模型信息: {model_info}")
            
            return True
        else:
            print(f"❌ k值文件不存在: {k_file}")
            return False
            
    except Exception as e:
        print(f"❌ 读取k值失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试新功能...")
    
    # 测试天气API
    weather_success = test_weather_api()
    
    # 测试模型k值
    k_value_success = test_model_k_value()
    
    print(f"\n📋 测试结果:")
    print(f"  天气API: {'✅ 通过' if weather_success else '❌ 失败'}")
    print(f"  模型k值: {'✅ 通过' if k_value_success else '❌ 失败'}")
    
    if weather_success and k_value_success:
        print("\n🎉 所有测试通过！应用已准备就绪。")
    else:
        print("\n⚠️  部分功能可能需要检查。")

if __name__ == "__main__":
    main() 