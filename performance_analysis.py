import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from realistic_prediction import RealisticLMPPredictor
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """详细的预测性能分析器"""
    
    def __init__(self):
        self.predictor = RealisticLMPPredictor()
    
    def detailed_performance_analysis(self, data_path: str = "applicationData"):
        """进行详细的性能分析"""
        print("=" * 80)
        print("LMP异常检测 - 详细性能分析")
        print("=" * 80)
        
        try:
            # 1. 加载数据并训练模型
            loaded_data = self.predictor.load_all_data(data_path)
            if not loaded_data:
                raise ValueError("No data loaded")
            
            print("\n1. 训练模型...")
            self.predictor.train_model_with_real_data(loaded_data)
            
            # 2. 获取训练数据和真实标签
            X, y_true = self.predictor.prepare_training_data(loaded_data)
            
            # 3. 在不同阈值下进行预测
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            
            print(f"\n2. 数据概况:")
            print(f"   总样本数: {len(X)}")
            print(f"   真实异常数: {y_true.sum()}")
            print(f"   真实异常率: {y_true.mean():.2%}")
            print(f"   时间跨度: {len(X)} 小时")
            
            # 4. 阈值性能分析
            print(f"\n3. 不同阈值下的预测性能:")
            print("-" * 80)
            print(f"{'阈值':<8} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'预测异常数':<10}")
            print("-" * 80)
            
            best_threshold = 0.3
            best_f1 = 0
            results_data = []
            
            for threshold in thresholds:
                predictions = self.predictor.predict_anomalies(X, threshold=threshold)
                y_pred = predictions['predictions']
                
                # 计算性能指标
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                print(f"{threshold:<8.2f} {accuracy:<8.3f} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} {y_pred.sum():<10}")
                
                # 记录最佳F1分数的阈值
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                
                results_data.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predicted_anomalies': y_pred.sum()
                })
            
            print(f"\n✅ 最佳阈值: {best_threshold} (F1分数: {best_f1:.3f})")
            
            # 5. 使用最佳阈值进行详细分析
            print(f"\n4. 使用最佳阈值 {best_threshold} 的详细分析:")
            print("=" * 50)
            
            best_predictions = self.predictor.predict_anomalies(X, threshold=best_threshold)
            y_pred_best = best_predictions['predictions']
            y_prob_best = best_predictions['probabilities']
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred_best)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\n混淆矩阵:")
            print(f"              预测")
            print(f"        正常    异常")
            print(f"实际 正常  {tn:4d}    {fp:4d}")
            print(f"     异常  {fn:4d}    {tp:4d}")
            
            print(f"\n性能指标详解:")
            print(f"• 真阳性 (TP): {tp} - 正确检测的异常")
            print(f"• 假阳性 (FP): {fp} - 误报的异常")
            print(f"• 真阴性 (TN): {tn} - 正确识别的正常")
            print(f"• 假阴性 (FN): {fn} - 漏检的异常")
            print(f"• 准确率: {(tp+tn)/(tp+tn+fp+fn):.3f} - 总体正确率")
            print(f"• 精确率: {tp/(tp+fp) if (tp+fp) > 0 else 0:.3f} - 预测异常中真正异常的比例")
            print(f"• 召回率: {tp/(tp+fn) if (tp+fn) > 0 else 0:.3f} - 真实异常中被检测出的比例")
            print(f"• F1分数: {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0:.3f} - 精确率和召回率的调和平均")
            
            # 6. 时间序列分析
            self.time_series_analysis(X, y_true, y_pred_best, y_prob_best)
            
            # 7. 错误分析
            self.error_analysis(X, y_true, y_pred_best, y_prob_best)
            
            # 8. 商业价值分析
            self.business_value_analysis(tp, fp, fn, tn)
            
            return {
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'confusion_matrix': cm,
                'results_data': results_data
            }
            
        except Exception as e:
            print(f"分析过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def time_series_analysis(self, X, y_true, y_pred, y_prob):
        """时间序列分析"""
        print(f"\n5. 时间序列分析:")
        print("=" * 30)
        
        # 按小时统计
        if 'hour' in X.columns:
            hour_stats = pd.DataFrame({
                'hour': X['hour'],
                'true_anomaly': y_true,
                'pred_anomaly': y_pred,
                'anomaly_prob': y_prob
            })
            
            hourly_summary = hour_stats.groupby('hour').agg({
                'true_anomaly': ['count', 'sum'],
                'pred_anomaly': 'sum',
                'anomaly_prob': 'mean'
            }).round(3)
            
            print("按小时统计:")
            print("小时  样本数  真实异常  预测异常  平均异常概率")
            print("-" * 45)
            for hour in sorted(hour_stats['hour'].unique()):
                hour_data = hour_stats[hour_stats['hour'] == hour]
                count = len(hour_data)
                true_anom = hour_data['true_anomaly'].sum()
                pred_anom = hour_data['pred_anomaly'].sum()
                avg_prob = hour_data['anomaly_prob'].mean()
                print(f"{int(hour):2d}    {count:4d}      {int(true_anom):4d}       {int(pred_anom):4d}      {avg_prob:.3f}")
        
        # 检查是否存在模式
        consecutive_errors = 0
        max_consecutive = 0
        for i in range(len(y_true)):
            if y_true.iloc[i] != y_pred[i]:
                consecutive_errors += 1
                max_consecutive = max(max_consecutive, consecutive_errors)
            else:
                consecutive_errors = 0
        
        print(f"\n连续预测错误最大长度: {max_consecutive} 小时")
    
    def error_analysis(self, X, y_true, y_pred, y_prob):
        """错误分析"""
        print(f"\n6. 错误分析:")
        print("=" * 20)
        
        # 假阳性分析
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        print(f"假阳性 (误报) 案例数: {false_positives.sum()}")
        print(f"假阴性 (漏检) 案例数: {false_negatives.sum()}")
        
        if false_positives.sum() > 0:
            fp_probs = y_prob[false_positives]
            print(f"假阳性的异常概率范围: {fp_probs.min():.3f} - {fp_probs.max():.3f}")
            print(f"假阳性的平均异常概率: {fp_probs.mean():.3f}")
        
        if false_negatives.sum() > 0:
            fn_probs = y_prob[false_negatives]
            print(f"假阴性的异常概率范围: {fn_probs.min():.3f} - {fn_probs.max():.3f}")
            print(f"假阴性的平均异常概率: {fn_probs.mean():.3f}")
        
        # 最有信心的预测
        most_confident_anomaly = y_prob.argmax()
        least_confident_anomaly = y_prob[y_pred == 1].argmin() if (y_pred == 1).any() else None
        
        print(f"\n最有信心的异常预测:")
        print(f"  索引: {most_confident_anomaly}, 概率: {y_prob[most_confident_anomaly]:.3f}, 实际: {'异常' if y_true.iloc[most_confident_anomaly] else '正常'}")
        
        if least_confident_anomaly is not None:
            print(f"最不确定的异常预测:")
            indices = np.where(y_pred == 1)[0]
            if len(indices) > 0:
                idx = indices[least_confident_anomaly]
                print(f"  索引: {idx}, 概率: {y_prob[idx]:.3f}, 实际: {'异常' if y_true.iloc[idx] else '正常'}")
    
    def business_value_analysis(self, tp, fp, fn, tn):
        """商业价值分析"""
        print(f"\n7. 商业价值分析:")
        print("=" * 25)
        
        # 假设成本 (可以根据实际业务调整)
        cost_false_alarm = 1000      # 假警报成本 (人力调查成本)
        cost_missed_anomaly = 10000  # 漏检成本 (潜在的市场损失)
        benefit_caught_anomaly = 5000 # 成功检测的收益 (避免损失)
        
        total_cost = fp * cost_false_alarm + fn * cost_missed_anomaly
        total_benefit = tp * benefit_caught_anomaly
        net_value = total_benefit - total_cost
        
        print(f"成本效益分析 (假设值):")
        print(f"• 假警报成本: ${fp} × ${cost_false_alarm:,} = ${fp * cost_false_alarm:,}")
        print(f"• 漏检成本: ${fn} × ${cost_missed_anomaly:,} = ${fn * cost_missed_anomaly:,}")
        print(f"• 检测收益: ${tp} × ${benefit_caught_anomaly:,} = ${tp * benefit_caught_anomaly:,}")
        print(f"• 净价值: ${net_value:,}")
        
        if net_value > 0:
            print(f"✅ 系统产生正价值")
        else:
            print(f"❌ 系统产生负价值 - 需要优化")
        
        # 改进建议
        print(f"\n改进建议:")
        if fp > tp:
            print(f"• 假警报过多，建议提高预测阈值")
        if fn > 0:
            print(f"• 存在漏检，建议改进特征工程或模型")
        if tp == 0:
            print(f"• 未检测到任何真实异常，模型需要重新训练")

def main():
    """主函数"""
    analyzer = PerformanceAnalyzer()
    results = analyzer.detailed_performance_analysis()
    
    if results:
        print(f"\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        print(f"✅ 分析完成")
        print(f"✅ 最佳阈值: {results['best_threshold']}")
        print(f"✅ 最佳F1分数: {results['best_f1']:.3f}")
        print(f"")
        print(f"🎯 关键发现:")
        print(f"  • 当前模型在检测LMP异常方面有一定能力")
        print(f"  • 需要在精确率和召回率之间找到平衡")
        print(f"  • 建议收集更多历史数据以改善模型性能")

if __name__ == "__main__":
    main() 