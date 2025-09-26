#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版电力市场LMP异常检测预测应用
集成PJM数据自动化功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import json
import os
from pjm_data_automation import PJMDataAutomation
from power_market_anomaly_detection import load_model_for_prediction, apply_holiday_features, holidays
import plotly.graph_objects as go
import plotly.express as px

# 设置页面配置
st.set_page_config(
    page_title="增强版电力市场LMP异常检测预测系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_trained_model():
    """加载训练好的模型"""
    try:
        model, scaler, feature_columns, k_value = load_model_for_prediction()
        return model, scaler, feature_columns, k_value, True
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None, None, None, False

def load_prediction_input(input_file):
    """加载预测输入数据"""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ 加载预测输入失败: {e}")
        return None

def make_prediction_with_automation(model, scaler, feature_columns, prediction_input):
    """使用自动化数据进行预测"""
    try:
        pred_date = dt.datetime.strptime(prediction_input['prediction_date'], '%Y-%m-%d').date()
        pred_hour = prediction_input['prediction_hour']
        da_prices = prediction_input['historical_da_prices']
        rolling_features = prediction_input['rolling_features']
        
        # 创建预测时间
        pred_datetime = dt.datetime.combine(pred_date, dt.time(pred_hour))
        
        # 创建输入数据DataFrame
        input_data = pd.DataFrame({
            'datetime_beginning_utc': [pred_datetime],
            'hour': [pred_hour],
            'day_of_week': [pred_date.weekday()],
            'month': [pred_date.month],
            'day_of_year': [pred_date.timetuple().tm_yday],
        })
        
        # 添加小时one-hot编码
        for h in range(24):
            input_data[f'hour_{h}'] = 1 if pred_hour == h else 0
        
        # 添加星期one-hot编码
        for d in range(7):
            input_data[f'dow_{d}'] = 1 if pred_date.weekday() == d else 0
        
        # 添加历史DA价格lag特征
        for i, price in enumerate(da_prices):
            input_data[f'total_lmp_da_lag_{i+1}'] = price
        
        # 添加滚动窗口特征
        windows = [3, 6, 12, 24]
        for w in windows:
            input_data[f'da_roll_mean_{w}h'] = rolling_features[f'roll_{w}h_mean']
            input_data[f'da_roll_std_{w}h'] = rolling_features[f'roll_{w}h_std']
        
        # 添加聚合的天气特征（使用默认值，因为PJM数据中没有天气）
        weather_features = {
            'agg_temp_mean': 20.0,
            'agg_temp_max': 25.0,
            'agg_temp_min': 15.0,
            'agg_temp_std': 3.0,
            'agg_wind_mean': 15.0,
            'agg_wind_max': 20.0,
            'agg_pressure_mean': 1013.25,
            'agg_soil_temp_mean': 18.0,
            'agg_soil_moisture_mean': 0.3
        }
        
        for key, value in weather_features.items():
            input_data[key] = value
        
        # 添加节假日特征
        input_data = apply_holiday_features(input_data, holidays)
        
        # 准备特征矩阵
        X_pred = pd.DataFrame()
        for col in feature_columns:
            if col in input_data.columns:
                X_pred[col] = input_data[col]
            else:
                X_pred[col] = 0
        
        # 标准化特征
        X_pred_scaled = scaler.transform(X_pred)
        
        # 进行预测
        prediction = model.predict(X_pred_scaled)[0]
        prediction_proba = model.predict_proba(X_pred_scaled)[0]
        
        return {
            'prediction': prediction,
            'probabilities': prediction_proba,
            'datetime': pred_datetime,
            'input_data': input_data,
            'da_prices': da_prices,
            'weather_features': weather_features,
            'rolling_features': rolling_features,
            'data_source': prediction_input['data_source'],
            'last_data_time': prediction_input['last_data_time']
        }
        
    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")
        return None

def display_enhanced_results(result):
    """显示增强版预测结果"""
    st.success("✅ 预测完成！")
    
    prediction = result['prediction']
    probabilities = result['probabilities']
    pred_datetime = result['datetime']
    
    # 获取预测标签和风险评估
    prediction_label = get_prediction_label(prediction)
    risk_level, risk_color = get_risk_level(prediction, probabilities)
    recommendation = get_recommendation(prediction, probabilities)
    
    # 显示主要结果
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "预测结果",
            prediction_label,
            help="基于历史数据和当前条件的异常检测结果"
        )
    
    with col2:
        st.metric(
            "风险等级",
            risk_level,
            help="基于预测概率的风险评估"
        )
    
    with col3:
        confidence = max(probabilities) * 100
        st.metric(
            "预测置信度",
            f"{confidence:.1f}%",
            help="模型对预测结果的置信程度"
        )
    
    with col4:
        st.metric(
            "数据来源",
            "PJM自动化",
            help="使用PJM DataMiner自动获取的数据"
        )
    
    # 显示概率分布
    st.subheader("📊 异常检测概率分布")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 创建概率柱状图
        categories = ['正常', '高异常', '低异常']
        colors = ['green', 'red', 'blue']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=probabilities,
                marker_color=colors,
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="异常检测概率",
            xaxis_title="类别",
            yaxis_title="概率",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 详细概率")
        for i, (category, prob) in enumerate(zip(categories, probabilities)):
            color = colors[i]
            st.write(f"**{category}**: {prob:.3f} ({prob*100:.1f}%)")
        
        st.divider()
        st.subheader("⚠️ 风险评估")
        if risk_color == 'green':
            st.success(f"🟢 {risk_level}")
        elif risk_color == 'yellow':
            st.warning(f"🟡 {risk_level}")
        else:
            st.error(f"🔴 {risk_level}")
    
    # 显示数据摘要
    st.subheader("📋 数据摘要")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**预测时间**: {pred_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**历史DA价格**: ${result['da_prices'][0]:.1f} ~ ${result['da_prices'][-1]:.1f}")
        rolling = result['rolling_features']
        st.write(f"**3小时均值/标准差**: ${rolling['roll_3h_mean']:.1f} / ${rolling['roll_3h_std']:.1f}")
        st.write(f"**24小时均值/标准差**: ${rolling['roll_24h_mean']:.1f} / ${rolling['roll_24h_std']:.1f}")
    
    with col2:
        st.write(f"**数据来源**: {result['data_source']}")
        st.write(f"**最后数据时间**: {result['last_data_time']}")
        st.write(f"**预测时间**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        is_holiday = result['input_data']['isHoliday'].iloc[0]
        st.write(f"**是否节假日**: {'是' if is_holiday else '否'}")
    
    st.subheader("💡 交易建议")
    st.info(recommendation)

def get_prediction_label(prediction):
    """获取预测标签"""
    labels = {0: "正常", 1: "高异常", 2: "低异常"}
    return labels.get(prediction, "未知")

def get_risk_level(prediction, probabilities):
    """获取风险等级"""
    if prediction == 0:
        return "低风险", "green"
    elif prediction == 1:
        if probabilities[1] > 0.8:
            return "高风险", "red"
        else:
            return "中风险", "yellow"
    else:  # prediction == 2
        if probabilities[2] > 0.8:
            return "高风险", "red"
        else:
            return "中风险", "yellow"

def get_recommendation(prediction, probabilities):
    """获取交易建议"""
    if prediction == 0:
        return "🟢 市场运行正常，建议按照日前市场价格进行正常的电力交易规划。"
    elif prediction == 1:
        if probabilities[1] > 0.8:
            return "🔴 高异常警告！实时价格可能显著高于日前价格，强烈建议减少实时市场购电或增加售电。"
        else:
            return "🟡 中等风险，实时价格可能高于日前价格，建议关注供需平衡情况，考虑适当调整交易策略。"
    else:  # prediction == 2
        if probabilities[2] > 0.8:
            return "🔴 高异常警告！实时价格可能显著低于日前价格，强烈建议增加实时市场购电或减少售电。"
        else:
            return "🟡 中等风险，实时价格可能低于日前价格，建议关注供需平衡情况，考虑适当调整交易策略。"

def main():
    """主函数"""
    st.title("⚡ 增强版电力市场LMP异常检测预测系统")
    st.markdown("**集成PJM数据自动化功能**")
    
    # 加载模型
    model, scaler, feature_columns, k_value, model_loaded = load_trained_model()
    
    if not model_loaded:
        st.error("❌ 无法加载模型，请先训练并保存模型")
        st.info("💡 请运行以下命令来训练模型：`python power_market_anomaly_detection.py`")
        return
    
    # 显示模型信息
    with st.sidebar:
        st.header("🤖 模型信息")
        st.success("✅ 模型已加载")
        st.info(f"模型类型: XGBoost")
        st.info(f"异常检测阈值 (k值): {k_value}")
        st.info(f"特征数量: {len(feature_columns)}")
        
        st.header("📊 预测类别说明")
        st.write("🟢 **正常**: 实时价格与日前价格差异正常")
        st.write("🔴 **高异常**: 实时价格可能显著高于日前价格")
        st.write("🔵 **低异常**: 实时价格可能显著低于日前价格")
    
    # 创建主要布局
    tab1, tab2, tab3 = st.tabs(["🔮 自动化预测", "📊 手动预测", "ℹ️ 使用说明"])
    
    with tab1:
        st.header("🤖 自动化预测（推荐）")
        st.write("使用PJM DataMiner自动获取最新数据并进行预测")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📅 预测参数")
            
            pred_date = st.date_input(
                "预测日期",
                value=dt.date.today() + dt.timedelta(days=1),
                help="选择要预测的日期"
            )
            
            pred_hour = st.selectbox(
                "预测小时",
                options=list(range(24)),
                index=12,
                help="选择要预测的小时 (0-23)"
            )
        
        with col2:
            st.subheader("⚙️ 自动化选项")
            
            days_back = st.slider(
                "历史数据天数",
                min_value=3,
                max_value=14,
                value=7,
                help="获取多少天的历史数据来计算特征"
            )
            
            if st.button("🚀 开始自动化预测", type="primary", use_container_width=True):
                with st.spinner("正在获取PJM数据并计算特征..."):
                    try:
                        # 创建PJM自动化实例
                        pjm_auto = PJMDataAutomation()
                        
                        # 获取数据并创建预测输入
                        prediction_input = pjm_auto.create_prediction_input(pred_date, pred_hour)
                        
                        # 进行预测
                        result = make_prediction_with_automation(
                            model, scaler, feature_columns, prediction_input
                        )
                        
                        if result:
                            display_enhanced_results(result)
                        
                    except Exception as e:
                        st.error(f"❌ 自动化预测失败: {e}")
                        st.info("💡 请检查网络连接或PJM DataMiner服务状态")
    
    with tab2:
        st.header("✋ 手动预测")
        st.write("手动输入参数进行预测（原始功能）")
        
        # 这里可以集成原来的手动输入功能
        st.info("💡 手动预测功能请使用原始的 `lmp_prediction_app.py`")
    
    with tab3:
        st.header("ℹ️ 使用说明")
        
        st.markdown("""
        ### 📖 增强版预测系统使用指南
        
        #### 🚀 自动化预测（推荐）
        1. **选择预测参数**: 选择预测日期和小时
        2. **设置历史数据**: 选择获取多少天的历史数据
        3. **开始预测**: 点击"开始自动化预测"按钮
        4. **查看结果**: 系统会自动获取PJM数据并显示预测结果
        
        #### ✋ 手动预测
        - 使用原始的 `lmp_prediction_app.py` 进行手动输入预测
        
        #### 🔧 技术特点
        - **自动数据获取**: 从PJM DataMiner自动获取最新LMP数据
        - **特征自动计算**: 自动计算滚动窗口特征
        - **实时预测**: 基于最新市场数据进行预测
        - **数据可视化**: 提供详细的概率分布和风险评估
        
        #### ⚠️ 注意事项
        - 需要网络连接访问PJM DataMiner
        - 预测结果仅供参考，不构成投资建议
        - 建议结合其他市场信息综合判断
        """)

if __name__ == "__main__":
    main()
