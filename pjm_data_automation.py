#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PJM LMP数据自动化获取和处理脚本
自动从PJM DataMiner获取LMP数据并计算滚动窗口特征
"""

import pandas as pd
import numpy as np
import requests
import os
import datetime as dt
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

class PJMDataAutomation:
    def __init__(self, output_dir="pjm_data"):
        """初始化PJM数据自动化类"""
        self.output_dir = output_dir
        self.base_url = "https://dataminer2.pjm.com/feed/da_hrl_lmps"
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ 创建输出目录: {output_dir}")
    
    def get_pjm_data(self, start_date, end_date, retry_count=3):
        """
        从PJM DataMiner获取LMP数据
        
        Parameters:
        - start_date: 开始日期 (YYYY-MM-DD)
        - end_date: 结束日期 (YYYY-MM-DD)
        - retry_count: 重试次数
        """
        print(f"📥 正在获取PJM数据: {start_date} 到 {end_date}")
        
        for attempt in range(retry_count):
            try:
                # 构建请求参数
                params = {
                    'download': 'true',
                    'startdate': start_date,
                    'enddate': end_date
                }
                
                print(f"🌐 请求URL: {self.base_url}")
                print(f"📋 参数: {params}")
                
                # 发送请求
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # 检查响应内容
                if len(response.content) < 100:
                    raise ValueError("响应数据太少，可能没有数据")
                
                # 解析CSV数据
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                
                if df.empty:
                    raise ValueError("没有获取到数据")
                
                print(f"✅ 成功获取 {len(df)} 行数据")
                return df
                
            except Exception as e:
                print(f"⚠️ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < retry_count - 1:
                    print(f"⏳ 等待 {2 ** attempt} 秒后重试...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ 所有重试都失败了")
                    raise e
    
    def process_lmp_data(self, df):
        """
        处理LMP数据，计算滚动窗口特征
        
        Parameters:
        - df: 原始LMP数据DataFrame
        """
        print("🔄 处理LMP数据...")
        
        # 确保时间列是datetime类型
        if 'datetime_beginning_utc' in df.columns:
            df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
        elif 'datetime' in df.columns:
            df['datetime_beginning_utc'] = pd.to_datetime(df['datetime'])
        else:
            # 尝试找到时间列
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df['datetime_beginning_utc'] = pd.to_datetime(df[time_cols[0]])
            else:
                raise ValueError("找不到时间列")
        
        # 按时间排序
        df = df.sort_values('datetime_beginning_utc').reset_index(drop=True)
        
        # 找到LMP价格列
        lmp_cols = [col for col in df.columns if 'lmp' in col.lower() and 'da' in col.lower()]
        if not lmp_cols:
            lmp_cols = [col for col in df.columns if 'price' in col.lower()]
        
        if not lmp_cols:
            print("⚠️ 可用列:", df.columns.tolist())
            raise ValueError("找不到LMP价格列")
        
        lmp_col = lmp_cols[0]
        print(f"📊 使用LMP列: {lmp_col}")
        
        # 计算滚动窗口特征
        windows = [3, 6, 12, 24]
        rolling_features = {}
        
        for w in windows:
            # 使用shift(1)避免数据泄漏
            shifted_data = df[lmp_col].shift(1)
            rolling_features[f'roll_mean_{w}h'] = shifted_data.rolling(window=w).mean()
            rolling_features[f'roll_std_{w}h'] = shifted_data.rolling(window=w).std()
        
        # 添加滚动特征到DataFrame
        for key, values in rolling_features.items():
            df[key] = values
        
        # 添加时间特征
        df['hour'] = df['datetime_beginning_utc'].dt.hour
        df['day_of_week'] = df['datetime_beginning_utc'].dt.dayofweek
        df['month'] = df['datetime_beginning_utc'].dt.month
        df['day_of_year'] = df['datetime_beginning_utc'].dt.dayofyear
        
        print(f"✅ 数据处理完成，添加了 {len(rolling_features)} 个滚动窗口特征")
        return df
    
    def get_latest_features(self, days_back=7):
        """
        获取最新的滚动窗口特征用于预测
        
        Parameters:
        - days_back: 获取多少天前的数据
        """
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days_back)
        
        print(f"📅 获取数据范围: {start_date} 到 {end_date}")
        
        # 获取数据
        df = self.get_pjm_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # 处理数据
        df_processed = self.process_lmp_data(df)
        
        # 获取最新的特征
        latest_row = df_processed.iloc[-1]
        
        # 提取历史DA价格 (最近7天)
        lmp_cols = [col for col in df_processed.columns if 'lmp' in col.lower() and 'da' in col.lower()]
        if not lmp_cols:
            lmp_cols = [col for col in df_processed.columns if 'price' in col.lower()]
        
        lmp_col = lmp_cols[0]
        
        # 获取最近7天的价格
        recent_prices = df_processed[lmp_col].tail(7).tolist()
        
        # 获取滚动窗口特征
        rolling_features = {}
        windows = [3, 6, 12, 24]
        for w in windows:
            rolling_features[f'roll_{w}h_mean'] = latest_row[f'roll_mean_{w}h']
            rolling_features[f'roll_{w}h_std'] = latest_row[f'roll_std_{w}h']
        
        # 保存数据
        output_file = os.path.join(self.output_dir, f"pjm_data_{start_date}_{end_date}.csv")
        df_processed.to_csv(output_file, index=False)
        print(f"💾 数据已保存到: {output_file}")
        
        return {
            'recent_prices': recent_prices,
            'rolling_features': rolling_features,
            'latest_datetime': latest_row['datetime_beginning_utc'],
            'data_file': output_file
        }
    
    def create_prediction_input(self, pred_date=None, pred_hour=12):
        """
        创建预测输入数据
        
        Parameters:
        - pred_date: 预测日期，默认为明天
        - pred_hour: 预测小时，默认12点
        """
        if pred_date is None:
            pred_date = dt.date.today() + dt.timedelta(days=1)
        
        print(f"🔮 创建预测输入: {pred_date} {pred_hour}:00")
        
        # 获取最新特征
        features = self.get_latest_features()
        
        # 创建预测输入
        prediction_input = {
            'prediction_date': pred_date,
            'prediction_hour': pred_hour,
            'historical_da_prices': features['recent_prices'],
            'rolling_features': features['rolling_features'],
            'data_source': features['data_file'],
            'last_data_time': features['latest_datetime']
        }
        
        # 保存预测输入
        input_file = os.path.join(self.output_dir, f"prediction_input_{pred_date}_{pred_hour:02d}h.json")
        import json
        with open(input_file, 'w') as f:
            json.dump(prediction_input, f, indent=2, default=str)
        
        print(f"💾 预测输入已保存到: {input_file}")
        return prediction_input

def main():
    """主函数"""
    print("🚀 PJM LMP数据自动化处理")
    
    # 创建自动化实例
    pjm_auto = PJMDataAutomation()
    
    try:
        # 获取最新数据并创建预测输入
        prediction_input = pjm_auto.create_prediction_input()
        
        print("\n📊 获取的数据摘要:")
        print(f"  预测日期: {prediction_input['prediction_date']}")
        print(f"  预测小时: {prediction_input['prediction_hour']}")
        print(f"  历史DA价格: {prediction_input['historical_da_prices']}")
        print(f"  数据文件: {prediction_input['data_source']}")
        print(f"  最后数据时间: {prediction_input['last_data_time']}")
        
        print("\n📈 滚动窗口特征:")
        for key, value in prediction_input['rolling_features'].items():
            print(f"  {key}: {value:.2f}")
        
        print("\n✅ 自动化处理完成！")
        print("💡 现在您可以使用这些数据在预测应用中进行预测了")
        
    except Exception as e:
        print(f"❌ 自动化处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
