#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_trained_models(load_path="trained_models/"):
    """
    加载训练好的模型、标准化器和最优阈值
    """
    print("=== 加载训练好的模型 ===")
    
    models_dict = {}
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'SVC': 'svc_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl'
    }
    
    # 加载模型
    for model_name, filename in model_files.items():
        model_path = os.path.join(load_path, filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models_dict[model_name] = pickle.load(f)
            print(f"✅ {model_name} 模型已加载")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
    
    # 加载标准化器
    scaler_path = os.path.join(load_path, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ 标准化器已加载")
    else:
        print(f"⚠️ 标准化器文件不存在: {scaler_path}")
    
    # 加载最优阈值
    thresholds_path = os.path.join(load_path, 'optimal_thresholds.pkl')
    optimal_thresholds = None
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'rb') as f:
            optimal_thresholds = pickle.load(f)
        print(f"✅ 最优阈值已加载")
    else:
        print(f"⚠️ 最优阈值文件不存在: {thresholds_path}")
    
    return models_dict, scaler, optimal_thresholds

def validate_model_with_new_data(processed_data_path="processed_new_data/", 
                                 models_path="trained_models/"):
    """
    使用训练好的模型验证新数据
    """
    print("=== 使用训练模型验证新数据 ===")
    
    # 加载训练好的模型
    models_dict, scaler, optimal_thresholds = load_trained_models(models_path)
    
    if not models_dict or scaler is None:
        print("❌ 无法加载训练好的模型")
        return None
    
    # 加载处理好的新数据
    complete_data_file = os.path.join(processed_data_path, "processed_new_data_complete.pkl")
    
    if not os.path.exists(complete_data_file):
        print(f"❌ 没有找到完整数据文件: {complete_data_file}")
        return None
    
    all_results = {}
    all_predictions = []
    
    try:
        # 直接读取完整的处理数据
        new_data = pd.read_pickle(complete_data_file)
        print(f"\n📊 验证完整数据集: {len(new_data)} 条记录")
        
        if len(new_data) == 0:
            print(f"⚠️ 数据为空")
            return None
        
        # 使用与训练时完全相同的特征列表和顺序
        inputs = [
            'lagged_lmp_da', 'lagged_delta',
            'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
            'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
            'isHoliday',
            'total_lmp_da_lag_1', 'total_lmp_da_lag_2', 'total_lmp_da_lag_3',
            'total_lmp_da_lag_4', 'total_lmp_da_lag_5', 'total_lmp_da_lag_6', 'total_lmp_da_lag_7',
            'hour_10AM', 'hour_10PM', 'hour_11AM', 'hour_11PM', 'hour_12AM', 'hour_12PM',
            'hour_1AM', 'hour_1PM', 'hour_2AM', 'hour_2PM', 'hour_3AM', 'hour_3PM',
            'hour_4AM', 'hour_4PM', 'hour_5AM', 'hour_5PM', 'hour_6AM', 'hour_6PM',
            'hour_7AM', 'hour_7PM', 'hour_8AM', 'hour_8PM', 'hour_9AM', 'hour_9PM'
        ]
        
        # 检查特征是否存在，对于缺失的特征用0填充
        available_inputs = []
        for col in inputs:
            if col in new_data.columns:
                available_inputs.append(col)
            else:
                # 对于缺失的特征，添加默认值（主要是小时编码）
                if col.startswith('hour_'):
                    new_data[col] = 0
                    available_inputs.append(col)
                else:
                    print(f"⚠️ 缺失特征: {col}")
        
        print(f"可用特征: {len(available_inputs)}/{len(inputs)}")
        
        if len(available_inputs) < len(inputs) * 0.7:  # 至少70%的特征可用
            print(f"⚠️ 可用特征不足，跳过预测")
            return None
        
        # 使用与训练时相同的特征顺序
        X_new = new_data[inputs]  # 现在所有特征都应该存在
        
        # 检查是否有NaN值
        if X_new.isnull().any().any():
            print(f"⚠️ 检测到NaN值，用0填充")
            X_new = X_new.fillna(0)
        
        # 标准化特征
        X_new_scaled = scaler.transform(X_new)
        y_true = new_data['target_c']
        
        complete_results = {}
        
        # 使用每个模型进行预测
        for model_name, model in models_dict.items():
            try:
                # 获取预测概率
                y_pred_proba = model.predict_proba(X_new_scaled)[:, 1]
                
                # 使用最优阈值
                threshold = optimal_thresholds.get(model_name, 0.5) if optimal_thresholds else 0.5
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # 计算评估指标
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0
                
                complete_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'threshold': threshold
                }
                
                print(f"{model_name}: 准确率={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                print(f"  预测异常: {y_pred.sum()}/{len(y_pred)} ({y_pred.mean()*100:.1f}%)")
                print(f"  真实异常: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)")
                
                # 保存预测结果
                prediction_record = {
                    'dataset': 'complete',
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'threshold': threshold,
                    'num_samples': len(y_true),
                    'num_anomalies_true': y_true.sum(),
                    'num_anomalies_pred': y_pred.sum()
                }
                all_predictions.append(prediction_record)
                
            except Exception as e:
                print(f"❌ {model_name} 预测失败: {e}")
        
        all_results['complete'] = complete_results
        
    except Exception as e:
        print(f"❌ 处理完整数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 保存验证结果
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"model_validation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 验证结果已保存: {results_file}")
        
        # 显示总体结果摘要
        print(f"\n📈 验证结果摘要:")
        summary = results_df.groupby('model').agg({
            'accuracy': 'mean',
            'precision': 'mean', 
            'recall': 'mean',
            'f1': 'mean',
            'auc': 'mean'
        }).round(3)
        print(summary)
    
    return all_results

if __name__ == "__main__":
    # 运行模型验证
    print("🚀 开始模型验证流程...")
    
    validation_results = validate_model_with_new_data()
    
    if validation_results:
        print(f"\n🎯 模型验证完成！")
        print("- 已使用训练好的模型对新数据进行预测")
        print("- 验证结果已保存到CSV文件")
        print("- 可以查看各模型在新数据上的表现")
    else:
        print("❌ 模型验证失败")
    
    print("\n💡 说明：")
    print("- 这个流程使用了之前训练好的4个模型")
    print("- 对25条新的LMP数据进行异常检测预测")
    print("- 输出各模型的准确率、F1分数等指标") 