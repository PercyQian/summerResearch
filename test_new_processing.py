#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from datetime import datetime

# 历史基准异常检测函数
def build_historical_baseline(historical_lmp_path="hourlyLmpData/westernData", 
                             historical_weather_path="hourlyWeatherData/openMeteo", 
                             k=2.5):
    """
    使用历史2-3年数据建立异常检测基准
    """
    print("=== 建立历史数据异常检测基准 ===")
    
    # 检查历史数据是否存在
    if not os.path.exists(historical_lmp_path):
        print(f"历史LMP数据路径不存在: {historical_lmp_path}")
        return None
        
    if not os.path.exists(historical_weather_path):
        print(f"历史天气数据路径不存在: {historical_weather_path}")
        return None
    
    # 检查天气数据文件格式
    weather_files = os.listdir(historical_weather_path)
    weather_files = [f for f in weather_files if f.startswith('weather_data_')]
    
    # 判断文件格式
    has_csv = any(f.endswith('.csv') for f in weather_files)
    has_pkl = any(f.endswith('.pkl') for f in weather_files)
    
    print(f"天气数据文件: CSV={has_csv}, PKL={has_pkl}")
    weather_ext = '.csv' if has_csv else '.pkl'
    
    # 合并所有历史数据
    all_historical_data = []
    lmp_files = os.listdir(historical_lmp_path)
    processed_count = 0
    
    for lmp_file in lmp_files:
        if lmp_file.startswith('lmp_data_') and lmp_file.endswith('.csv'):
            hour = lmp_file.replace('lmp_data_', '').replace('.csv', '')
            
            # 读取LMP数据
            lmp_path = os.path.join(historical_lmp_path, lmp_file)
            
            try:
                lmp_data = pd.read_pickle(lmp_path)
                print(f"📁 {hour}: 读取pickle格式LMP文件 ({len(lmp_data)} 条记录)")
            except:
                try:
                    lmp_data = pd.read_csv(lmp_path)
                    print(f"📁 {hour}: 读取CSV格式LMP文件 ({len(lmp_data)} 条记录)")
                except Exception as e:
                    print(f"❌ 无法读取LMP文件 {lmp_file}: {e}")
                    continue
            
            # 读取对应的天气数据
            weather_file = f"weather_data_{hour}{weather_ext}"
            weather_path = os.path.join(historical_weather_path, weather_file)
            
            if os.path.exists(weather_path):
                try:
                    try:
                        weather_data = pd.read_pickle(weather_path)
                        print(f"🌤️ {hour}: 读取pickle格式天气文件 ({len(weather_data)} 条记录)")
                    except:
                        weather_data = pd.read_csv(weather_path)
                        print(f"🌤️ {hour}: 读取CSV格式天气文件 ({len(weather_data)} 条记录)")
                    
                    # 简化的数据合并（不依赖复杂的combineDataFrames函数）
                    min_len = min(len(lmp_data), len(weather_data))
                    combined = lmp_data[:min_len].copy()
                    
                    # 添加天气特征
                    weather_cols = ['apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
                                   'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)']
                    for col in weather_cols:
                        if col in weather_data.columns:
                            combined[col] = weather_data[col][:min_len].values
                    
                    combined['hour'] = hour
                    
                    # 计算误差
                    combined['error'] = combined['total_lmp_rt'] - combined['total_lmp_da']
                    
                    # 提取时间特征
                    if combined['datetime_beginning_utc'].dtype == 'object':
                        combined['datetime_beginning_utc'] = pd.to_datetime(combined['datetime_beginning_utc'], format='%m/%d/%Y %I:%M:%S %p')
                    
                    combined['hour_num'] = combined['datetime_beginning_utc'].dt.hour
                    combined['dow'] = combined['datetime_beginning_utc'].dt.dayofweek
                    
                    all_historical_data.append(combined)
                    processed_count += 1
                    print(f"✅ 已处理 {hour}: {len(combined)} 条记录")
                    
                except Exception as e:
                    print(f"❌ 读取天气文件失败 {weather_file}: {e}")
                    continue
                    
            else:
                print(f"⚠️ 未找到对应天气文件: {weather_path}")
    
    print(f"\n📊 处理结果: 成功处理了 {processed_count} 个时段的数据")
    
    if not all_historical_data:
        print("❌ 没有找到有效的历史数据")
        return None
    
    # 合并所有历史数据
    print("合并历史数据...")
    historical_df = pd.concat(all_historical_data, axis=0, ignore_index=True)
    
    print(f"✅ 历史数据总计: {len(historical_df)} 条记录")
    print(f"时间跨度: {historical_df['datetime_beginning_utc'].min()} 到 {historical_df['datetime_beginning_utc'].max()}")
    print(f"误差统计: 均值={historical_df['error'].mean():.3f}, 标准差={historical_df['error'].std():.3f}")
    
    # 计算每个时段的统计基准
    print("计算时段基准统计量...")
    baseline_stats = historical_df.groupby(['hour_num', 'dow'])['error'].agg(
        mu='mean',
        sigma='std', 
        count='count'
    ).reset_index()
    
    # 处理标准差为NaN或0的情况
    global_sigma = historical_df['error'].std()
    baseline_stats['sigma'] = baseline_stats['sigma'].fillna(global_sigma)
    baseline_stats.loc[baseline_stats['sigma'] == 0, 'sigma'] = global_sigma
    
    print(f"✅ 建立了 {len(baseline_stats)} 个时段基准")
    print("基准统计量示例:")
    print(baseline_stats.head(10))
    
    return baseline_stats

def detect_anomalies_with_baseline(new_data, baseline_stats, k=2.5):
    """
    使用历史基准检测新数据中的异常
    """
    print(f"=== 使用历史基准检测新数据异常 (k={k}) ===")
    
    df = new_data.copy()
    
    # 计算误差
    df['error'] = df['total_lmp_rt'] - df['total_lmp_da']
    
    # 提取时间特征
    try:
        if df['datetime_beginning_utc'].dtype == 'object':
            df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
        
        df['hour_num'] = df['datetime_beginning_utc'].dt.hour
        df['dow'] = df['datetime_beginning_utc'].dt.dayofweek
    except Exception as e:
        print(f"⚠️ 时间解析失败，使用hourTime提取小时: {e}")
        if 'hourTime' in df.columns:
            hour_mapping = {
                '12AM': 0, '1AM': 1, '2AM': 2, '3AM': 3, '4AM': 4, '5AM': 5, '6AM': 6, '7AM': 7, '8AM': 8, '9AM': 9, '10AM': 10, '11AM': 11,
                '12PM': 12, '1PM': 13, '2PM': 14, '3PM': 15, '4PM': 16, '5PM': 17, '6PM': 18, '7PM': 19, '8PM': 20, '9PM': 21, '10PM': 22, '11PM': 23
            }
            df['hour_num'] = df['hourTime'].map(hour_mapping).fillna(0).astype(int)
            df['dow'] = 0  # 假设为周一
    
    print(f"新数据: {len(df)} 条记录")
    print(f"时间跨度: {df['datetime_beginning_utc'].min()} 到 {df['datetime_beginning_utc'].max()}")
    
    # 合并基准统计量
    df = df.merge(baseline_stats, on=['hour_num', 'dow'], how='left')
    
    # 对于没有历史基准的时段，使用全局统计量
    global_mu = baseline_stats['mu'].mean()
    global_sigma = baseline_stats['sigma'].mean()
    
    df['mu'] = df['mu'].fillna(global_mu)
    df['sigma'] = df['sigma'].fillna(global_sigma)
    
    # 计算异常
    df['abs_deviation'] = np.abs(df['error'] - df['mu'])
    df['threshold'] = k * df['sigma']
    df['is_anomaly'] = (df['abs_deviation'] > df['threshold']).astype(int)
    
    # 统计结果
    total_anomalies = df['is_anomaly'].sum()
    anomaly_rate = total_anomalies / len(df) * 100
    
    print(f"\n异常检测结果:")
    print(f"总异常数量: {total_anomalies}")
    print(f"异常率: {anomaly_rate:.2f}%")
    
    # 显示异常详情
    if total_anomalies > 0:
        anomalies = df[df['is_anomaly'] == 1].sort_values('abs_deviation', ascending=False)
        print(f"\n前{min(5, len(anomalies))}个最大异常:")
        cols = ['datetime_beginning_utc', 'error', 'abs_deviation', 'threshold', 'hour_num', 'dow']
        print(anomalies[cols].head(5))
    else:
        print("\n没有检测到异常")
    
    return df

def total_lmp_delta(data):
    """计算LMP偏差"""
    data['total_lmp_delta'] = 0
    for i in list(data.index):
        a = data.loc[i, "total_lmp_da"]
        r = data.loc[i, "total_lmp_rt"]
        if abs(a) < 1e-6:
            a = 1e-6
        data.loc[i, "total_lmp_delta"] = abs((a - r) / a)

def applyHoliday(data, holidays):
    """添加假日标记"""
    data['isHoliday'] = 0  # 简化版：全部设为非假日
    return data

def process_new_lmp_data_complete(da_file, rt_file, weather_file=None, output_path="processed_new_data/"):
    """
    处理新的LMP数据（完整版：直接处理时间序列数据，类似训练时的all_data）
    """
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("=== 处理新的LMP数据（完整版）===")
    
    # 读取LMP数据
    try:
        da_data = pd.read_csv(da_file)
        rt_data = pd.read_csv(rt_file)
        print(f"DA数据: {len(da_data)} 行")
        print(f"RT数据: {len(rt_data)} 行")
    except Exception as e:
        print(f"❌ 读取LMP数据失败: {e}")
        return None
    
    # 合并DA和RT数据
    if len(da_data) != len(rt_data):
        print("⚠️ DA和RT数据行数不匹配，取较小长度")
        min_len = min(len(da_data), len(rt_data))
        da_data = da_data[:min_len].reset_index(drop=True)
        rt_data = rt_data[:min_len].reset_index(drop=True)
    
    # 合并DA和RT数据
    combined_lmp = da_data.copy()
    if 'total_lmp_rt' not in combined_lmp.columns:
        rt_col = 'total_lmp_rt' if 'total_lmp_rt' in rt_data.columns else rt_data.columns[-1]
        combined_lmp['total_lmp_rt'] = rt_data[rt_col]
    
    print(f"合并后LMP数据: {len(combined_lmp)} 行")
    print(f"数据列: {combined_lmp.columns.tolist()}")
    
    # 从时间列提取小时信息并创建one-hot编码（类似训练时的处理）
    print("提取小时信息并创建编码...")
    
    # 解析时间并提取小时
    try:
        # 尝试解析时间格式
        combined_lmp['datetime_parsed'] = pd.to_datetime(combined_lmp['datetime_beginning_utc'])
        combined_lmp['hour_24'] = combined_lmp['datetime_parsed'].dt.hour
        
        # 转换为12小时制格式（与训练时一致）
        def convert_to_12h_format(hour_24):
            if hour_24 == 0:
                return "12AM"
            elif hour_24 < 12:
                return f"{hour_24}AM"
            elif hour_24 == 12:
                return "12PM"
            else:
                return f"{hour_24-12}PM"
        
        combined_lmp['hourTime'] = combined_lmp['hour_24'].apply(convert_to_12h_format)
        
        # 显示小时分布
        hour_counts = combined_lmp['hourTime'].value_counts()
        print(f"小时分布: {hour_counts.to_dict()}")
        
    except Exception as e:
        print(f"⚠️ 时间解析失败: {e}")
        # 如果解析失败，使用默认值
        combined_lmp['hourTime'] = "4AM"  # 默认小时
    
    # 添加小时one-hot编码（与训练时完全一致）
    print("创建小时one-hot编码...")
    all_hours = ['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM',
                '12PM', '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM']
    
    # 使用pandas get_dummies创建one-hot编码（更高效）
    hour_dummies = pd.get_dummies(combined_lmp['hourTime'], prefix='hour')
    
    # 确保所有24小时都有对应的列（即使数据中没有）
    for h in all_hours:
        col_name = f'hour_{h}'
        if col_name not in hour_dummies.columns:
            hour_dummies[col_name] = 0
    
    # 按正确顺序排列小时列
    hour_cols = [f'hour_{h}' for h in all_hours]
    combined_lmp = pd.concat([combined_lmp, hour_dummies[hour_cols]], axis=1)
    
    # 添加天气特征（使用实际天气数据）
    print("添加天气特征...")
    
    if weather_file and os.path.exists(weather_file):
        try:
            # 读取天气数据
            if weather_file.endswith('.pkl'):
                weather_data = pd.read_pickle(weather_file)
            else:
                weather_data = pd.read_csv(weather_file)
            
            print(f"加载天气数据: {len(weather_data)} 行")
            
            # 解析天气数据的时间
            weather_data['datetime_parsed'] = pd.to_datetime(weather_data['time'])
            weather_data['datetime_rounded'] = weather_data['datetime_parsed'].dt.round('H')
            
            # 解析LMP数据的时间并四舍五入到小时
            combined_lmp['datetime_rounded'] = pd.to_datetime(combined_lmp['datetime_beginning_utc']).dt.round('H')
            
            # 按时间匹配天气数据
            weather_cols = [
                'apparent_temperature (°C)', 
                'wind_gusts_10m (km/h)', 
                'pressure_msl (hPa)',
                'soil_temperature_0_to_7cm (°C)', 
                'soil_moisture_0_to_7cm (m³/m³)'
            ]
            
            # 创建天气数据的查找字典
            weather_lookup = {}
            for _, row in weather_data.iterrows():
                time_key = row['datetime_rounded']
                weather_lookup[time_key] = {col: row[col] for col in weather_cols if col in row}
            
            # 为每行LMP数据匹配天气数据
            for col in weather_cols:
                combined_lmp[col] = 0  # 初始化
            
            matched_count = 0
            for idx, row in combined_lmp.iterrows():
                time_key = row['datetime_rounded']
                if time_key in weather_lookup:
                    for col in weather_cols:
                        if col in weather_lookup[time_key]:
                            combined_lmp.loc[idx, col] = weather_lookup[time_key][col]
                    matched_count += 1
                else:
                    # 如果没有精确匹配，使用最近的天气数据
                    closest_time = min(weather_lookup.keys(), key=lambda x: abs((x - time_key).total_seconds()))
                    for col in weather_cols:
                        if col in weather_lookup[closest_time]:
                            combined_lmp.loc[idx, col] = weather_lookup[closest_time][col]
            
            print(f"✅ 天气特征添加完成：{matched_count}/{len(combined_lmp)} 条记录精确匹配")
            
        except Exception as e:
            print(f"⚠️ 读取天气数据失败: {e}")
            print("使用默认天气值...")
            # 使用默认值作为备选
            default_weather_values = {
                'apparent_temperature (°C)': 15.0,
                'wind_gusts_10m (km/h)': 20.0,
                'pressure_msl (hPa)': 1013.25,
                'soil_temperature_0_to_7cm (°C)': 12.0,
                'soil_moisture_0_to_7cm (m³/m³)': 0.3
            }
            for col, val in default_weather_values.items():
                combined_lmp[col] = val
    else:
        print("⚠️ 没有提供天气文件，使用默认值")
        # 使用默认值
        default_weather_values = {
            'apparent_temperature (°C)': 15.0,
            'wind_gusts_10m (km/h)': 20.0,
            'pressure_msl (hPa)': 1013.25,
            'soil_temperature_0_to_7cm (°C)': 12.0,
            'soil_moisture_0_to_7cm (m³/m³)': 0.3
        }
        for col, val in default_weather_values.items():
            combined_lmp[col] = val
    
    # 添加其他特征
    print("添加其他特征...")
    
    # 假日特征
    combined_lmp = applyHoliday(combined_lmp, [])
    
    # 计算LMP偏差
    total_lmp_delta(combined_lmp)
    
    # 添加滞后特征（与训练时一致）
    combined_lmp['lagged_lmp_da'] = combined_lmp['total_lmp_da'].shift(1).fillna(combined_lmp['total_lmp_da'].iloc[0])
    combined_lmp['lagged_lmp_rt'] = combined_lmp['total_lmp_rt'].shift(1).fillna(combined_lmp['total_lmp_rt'].iloc[0])
    combined_lmp['lagged_delta'] = combined_lmp['total_lmp_delta'].shift(1).fillna(0)
    
    # 添加多个滞后特征（如果训练时使用了）
    k = 7  # 与训练时一致
    for i in range(1, k+1):
        combined_lmp[f'total_lmp_da_lag_{i}'] = combined_lmp['total_lmp_da'].shift(i).fillna(combined_lmp['total_lmp_da'].iloc[0])
    
    # 使用历史基准的异常检测（基于时间和星期的相对差异）
    print("使用历史基准检测异常...")
    try:
        # 建立历史基准
        baseline_stats = build_historical_baseline(k=2.5)
        
        if baseline_stats is not None:
            # 使用历史基准检测异常
            combined_lmp = detect_anomalies_with_baseline(combined_lmp, baseline_stats, k=2.5)
            
            # 将异常检测结果转换为target_c格式
            combined_lmp['target_c'] = combined_lmp['is_anomaly'].astype(int)
            combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
            
            print(f"基于历史基准的异常分布: {combined_lmp['target_c'].value_counts().to_dict()}")
        else:
            print("⚠️ 无法建立历史基准，使用简化的异常检测")
            # 基于当前数据的简化异常检测（备选方案）
            if len(combined_lmp) > 10:
                # 使用更严格的阈值（99%分位数而不是95%）
                threshold_99 = combined_lmp['total_lmp_delta'].quantile(0.99)
                combined_lmp['target_c'] = (combined_lmp['total_lmp_delta'] >= threshold_99).astype(int)
                print(f"使用99%分位数阈值: {threshold_99:.4f}")
            else:
                combined_lmp['target_c'] = 0  # 数据太少，默认为非异常
            combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
            print(f"简化异常分布: {combined_lmp['target_c'].value_counts().to_dict()}")
            
    except Exception as e:
        print(f"⚠️ 历史基准异常检测失败: {e}")
        # fallback到最简单的方法
        combined_lmp['target_c'] = 0  # 默认为非异常
        combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
        print("使用默认异常标记（全部为非异常）")
    
    # 保存处理后的完整数据
    output_file = os.path.join(output_path, "processed_new_data_complete.pkl")
    combined_lmp.to_pickle(output_file)
    print(f"💾 完整数据已保存: {output_file} ({len(combined_lmp)} 条记录)")
    
    print(f"\n✅ 新数据处理完成: {len(combined_lmp)} 条记录")
    print(f"包含特征: {len(combined_lmp.columns)} 列")
    
    # 显示特征列表
    feature_cols = [col for col in combined_lmp.columns if col.startswith(('lagged_', 'apparent_', 'wind_', 'pressure_', 'soil_', 'isHoliday', 'hour_', 'total_lmp_da_lag_'))]
    print(f"特征列: {len(feature_cols)} 个")
    print(f"前10个特征: {feature_cols[:10]}")
    
    # 返回字典格式以保持兼容性
    return {"complete": combined_lmp}

if __name__ == "__main__":
    # 测试新的数据处理流程
    result = process_new_lmp_data_complete(
        da_file="applicationData/da_hrl_lmps_2025.csv",
        rt_file="applicationData/rt_hrl_lmps_2025.csv",
        weather_file="weatherData/meteo/western_hub_weather_2025-01-01_to_2025-06-06.pkl"
    )
    
    if result:
        print("🎯 数据处理成功！")
        data = result["complete"]
        print(f"最终数据形状: {data.shape}")
        
        # 检查关键特征是否存在
        required_features = [
            'lagged_lmp_da', 'lagged_delta',
            'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
            'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
            'isHoliday', 'hour_4AM', 'total_lmp_da_lag_1'
        ]
        
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            print(f"⚠️ 缺失特征: {missing_features}")
        else:
            print("✅ 所有必需特征都存在")
    else:
        print("❌ 数据处理失败") 