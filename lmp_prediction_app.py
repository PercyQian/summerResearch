#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Market LMP Anomaly Detection Prediction App - Streamlit Version
Use trained models for real-time prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from power_market_anomaly_detection import load_model_for_prediction, apply_holiday_features, holidays
import os
import requests

# 设置页面配置
st.set_page_config(
    page_title="Power Market LMP Anomaly Detection Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 缓存模型加载
@st.cache_resource
def load_trained_model():
    """load trained models"""
    try:
        model, scaler, feature_columns, k_value = load_model_for_prediction()
        return model, scaler, feature_columns, k_value, True
    except Exception as e:
        return None, None, None, None, False

# zone coordinates (from original code)
zone_coords = {
    'AE':    (39.3643,  -74.4229),
    'BGE':   (39.2904,  -76.6122),
    'DPL':   (39.1582,  -75.5244),
    'JCPL':  (40.2171,  -74.7429),
    'METED': (40.3356,  -75.9269),
    'PECO':  (39.9526,  -75.1652),
    'PPL':   (40.6084,  -75.4902),
    'PEPCO': (38.9072,  -77.0369),
    'DOM':   (37.5407,  -77.4360),
    'APS':   (38.3498,  -81.6326),
    'PENELEC':(41.4089, -75.6624),
    'PSEG':  (40.7357,  -74.1724),
    'RECO':  (41.1486,  -73.9881),
}

@st.cache_data(ttl=3600)  # cache for 1 hour
def get_all_zones_weather_forecast(pred_date, pred_hour):
    """
    get weather forecast for all 13 zones and aggregate
    simulate the processing of original training data
    """
    try:
        target_date = pred_date.strftime('%Y-%m-%d')
        target_time = f"{target_date}T{pred_hour:02d}:00"
        
        all_weather_data = {}
        failed_zones = []
        
        # get weather data for all zones
        for zone_name, (lat, lon) in zone_coords.items():
            try:
                # Open-Meteo forecast API
                base_url = "https://api.open-meteo.com/v1/forecast"
                
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'hourly': [
                        'apparent_temperature', 
                        'pressure_msl',
                        'wind_gusts_10m',
                        'soil_temperature_0cm',
                        'soil_moisture_0_to_1cm'
                    ],
                    'timezone': 'America/New_York',
                    'start_date': target_date,
                    'end_date': target_date
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                hourly_data = data['hourly']
                times = hourly_data['time']
                
                if target_time in times:
                    idx = times.index(target_time)
                    
                    # use original column names
                    zone_weather = {
                        f'{zone_name}_apparent_temperature (°C)': hourly_data['apparent_temperature'][idx],
                        f'{zone_name}_pressure_msl (hPa)': hourly_data['pressure_msl'][idx],
                        f'{zone_name}_wind_gusts_10m (km/h)': hourly_data['wind_gusts_10m'][idx],
                        f'{zone_name}_soil_temperature_0_to_7cm (°C)': hourly_data['soil_temperature_0cm'][idx],
                        f'{zone_name}_soil_moisture_0_to_7cm (m³/m³)': hourly_data['soil_moisture_0_to_1cm'][idx]
                    }
                    
                    # filter None values
                    for key, value in zone_weather.items():
                        if value is not None:
                            all_weather_data[key] = value
                        
                else:
                    failed_zones.append(zone_name)
                    
            except Exception as e:
                st.warning(f"⚠️ get weather data for {zone_name} failed: {e}")
                failed_zones.append(zone_name)
        
        if not all_weather_data:
            st.error("❌ cannot get weather data for any zone")
            return None
            
        # aggregate weather data according to original data processing method
        weather_feature_cols = {
            'temp': [col for col in all_weather_data.keys() if 'apparent_temperature' in col],
            'wind': [col for col in all_weather_data.keys() if 'wind_gusts' in col],
            'pressure': [col for col in all_weather_data.keys() if 'pressure_msl' in col],
            'soil_temp': [col for col in all_weather_data.keys() if 'soil_temperature' in col],
            'soil_moisture': [col for col in all_weather_data.keys() if 'soil_moisture' in col]
        }
        
        aggregated_weather = {}
        
        # aggregate temperature
        temp_cols = weather_feature_cols['temp']
        if temp_cols:
            temp_values = [all_weather_data[col] for col in temp_cols if all_weather_data[col] is not None]
            if temp_values:
                aggregated_weather['agg_temp_mean'] = np.mean(temp_values)
                aggregated_weather['agg_temp_max'] = np.max(temp_values)
                aggregated_weather['agg_temp_min'] = np.min(temp_values)
                aggregated_weather['agg_temp_std'] = np.std(temp_values) if len(temp_values) > 1 else 0.0
        
        # aggregate wind speed
        wind_cols = weather_feature_cols['wind']
        if wind_cols:
            wind_values = [all_weather_data[col] for col in wind_cols if all_weather_data[col] is not None]
            if wind_values:
                aggregated_weather['agg_wind_mean'] = np.mean(wind_values)
                aggregated_weather['agg_wind_max'] = np.max(wind_values)
        
        # aggregate other features
        for short_name in ['pressure', 'soil_temp', 'soil_moisture']:
            cols = weather_feature_cols[short_name]
            if cols:
                values = [all_weather_data[col] for col in cols if all_weather_data[col] is not None]
                if values:
                    aggregated_weather[f'agg_{short_name}_mean'] = np.mean(values)
        
        # fill missing values
        default_values = {
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
        
        for key, default_val in default_values.items():
            if key not in aggregated_weather:
                aggregated_weather[key] = default_val
        
        # add successful zones information
        successful_zones = len(zone_coords) - len(failed_zones)
        aggregated_weather['successful_zones'] = successful_zones
        aggregated_weather['failed_zones'] = failed_zones
        aggregated_weather['source'] = 'API_all_zones'
        
        return aggregated_weather
        
    except Exception as e:
        st.error(f"❌ weather data aggregation failed: {e}")
        return None

# ===========================
# DA LMP helpers (PJM / CSV)
# ===========================

def _normalize_da_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded/downloaded DA LMP csv to two columns: datetime_beginning_utc, total_lmp_da."""
    cols = {c.lower(): c for c in df.columns}
    # try to locate datetime column
    dt_col = None
    for candidate in ['datetime_beginning_utc', 'datetime_beginning_ept', 'datetime_beginning_edt', 'datetime_beginning_est', 'delivery_hour']:  # common names
        if candidate in cols:
            dt_col = cols[candidate]
            break
    if dt_col is None:
        raise ValueError("cannot find datetime column in csv (expect datetime_beginning_utc/ept)")
    # try to locate DA total LMP column
    price_col = None
    for candidate in ['total_lmp_da', 'total_lmp', 'da_lmp']:
        if candidate in cols:
            price_col = cols[candidate]
            break
    if price_col is None:
        # PJM feed sometimes splits energy/congestion/loss, synthesize total if possible
        cand_energy = cols.get('da_energy_lmp') or cols.get('lmp_energy_da')
        cand_cong = cols.get('da_congestion_lmp') or cols.get('lmp_congestion_da')
        cand_loss = cols.get('da_loss_lmp') or cols.get('lmp_loss_da')
        if cand_energy and cand_cong and cand_loss:
            df['__total_lmp_da__'] = df[cand_energy].astype(float) + df[cand_cong].astype(float) + df[cand_loss].astype(float)
            price_col = '__total_lmp_da__'
        else:
            raise ValueError("cannot find total DA LMP column in csv (expect total_lmp_da or components)")
    out = df[[dt_col, price_col]].copy()
    out.rename(columns={dt_col: 'datetime_beginning_utc', price_col: 'total_lmp_da'}, inplace=True)
    out['datetime_beginning_utc'] = pd.to_datetime(out['datetime_beginning_utc'])
    out = out.sort_values('datetime_beginning_utc')
    return out

@st.cache_data(ttl=900)
def read_da_from_csv_url(csv_url: str) -> pd.DataFrame:
    """Read DA LMP from a direct CSV url (e.g. PJM Data Miner 2 export)."""
    df = pd.read_csv(csv_url)
    return _normalize_da_dataframe(df)

@st.cache_data(ttl=900)
def read_da_from_uploaded_files(files: list) -> pd.DataFrame:
    """Read and merge multiple uploaded CSV files into a single normalized DA dataframe."""
    all_parts = []
    for f in files:
        try:
            part = pd.read_csv(f)
            part = _normalize_da_dataframe(part)
            all_parts.append(part)
        except Exception as e:
            st.warning(f"⚠️ failed to parse {getattr(f, 'name', 'uploaded file')}: {e}")
    if not all_parts:
        raise ValueError("no valid csv parsed")
    df = pd.concat(all_parts, axis=0, ignore_index=True).sort_values('datetime_beginning_utc').drop_duplicates('datetime_beginning_utc')
    return df

def compute_da_features_for_prediction(da_df: pd.DataFrame, pred_datetime: pd.Timestamp) -> dict:
    """Compute 7-day lags and rolling windows exactly as training (shift(1) then rolling)."""
    series = da_df.set_index('datetime_beginning_utc')['total_lmp_da'].sort_index()
    # ensure hourly frequency for safe rolling
    series = series.asfreq('H')
    shifted = series.shift(1)
    feats = {}
    # lags: k days back at same hour
    for k in range(1, 8):
        ts = pred_datetime - pd.Timedelta(hours=24 * k)
        if ts not in series.index or pd.isna(series.get(ts)):
            raise ValueError(f"missing DA price for lag {k} at {ts}")
        feats[f'total_lmp_da_lag_{k}'] = float(series.loc[ts])
    # rolling windows up to pred time (exclusive) using shifted
    for w in [3, 6, 12, 24]:
        window_slice = shifted.loc[:pred_datetime].tail(w)
        feats[f'roll_{w}h_mean'] = float(window_slice.mean()) if len(window_slice) > 0 else 0.0
        feats[f'roll_{w}h_std'] = float(window_slice.std(ddof=0)) if len(window_slice) > 1 else 0.0
    return feats

def main():
    """main function"""
    st.title("⚡ Power Market LMP Anomaly Detection Prediction System")
    
    # load model
    model, scaler, feature_columns, k_value, model_loaded = load_trained_model()
    
    if not model_loaded:
        st.error("❌ cannot load model, please train and save model first")
        st.info("💡 please run the following command to train models: `python power_market_anomaly_detection.py`")
        return
    
    # display model information
    with st.sidebar:
        st.header("🤖 model information")
        st.success("✅ model loaded")
        st.info(f"model type: XGBoost")
        st.info(f"anomaly detection threshold (k value): {k_value}")
        st.info(f"number of features: {len(feature_columns)}")
        
        st.header("📊 prediction category explanation")
        st.write("🟢 **normal**: the difference between real-time price and day-ahead price is normal")
        st.write("🔴 **high anomaly**: the real-time price may significantly higher than the day-ahead price")
        st.write("🔵 **low anomaly**: the real-time price may significantly lower than the day-ahead price")
    
    # create main layout
    tab1, tab2, tab3 = st.tabs(["🔮 real-time prediction", "📊 historical analysis", "ℹ️ usage instructions"])
    
    with tab1:
        st.header("real-time anomaly detection prediction")
        
        # create two columns layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📅 basic parameters")
            
            # time input
            pred_date = st.date_input(
                "prediction date",
                value=dt.date.today() + dt.timedelta(days=1),
                help="select the date to predict"
            )
            
            pred_hour = st.selectbox(
                "prediction hour",
                options=list(range(24)),
                index=12,
                help="select the hour to predict (0-23)"
            )
            
            st.subheader("💰 historical DA price ($/MWh)")
            st.write("please input the day-ahead market price for the past 7 days:")
            
            # create historical price input
            da_prices = []
            for i in range(7):
                price = st.number_input(
                    f"day-ahead price for the past {i+1} days",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.get(f'prefill_da_price_{i}', 50.0),
                    step=1.0,
                    key=f"da_price_{i}"
                )
                da_prices.append(price)
            
            # Auto-fill from PJM Data Miner 2 / CSV
            with st.expander("📥 auto load DA from PJM Data Miner 2 or CSV", expanded=False):
                st.markdown("from PJM Data Miner 2 get CSV or upload local CSV, automatically calculate 7-day lags and rolling window features. Data source reference: [PJM Data Miner 2 - da_hrl_lmps](https://dataminer2.pjm.com/feed/da_hrl_lmps)")
                csv_url = st.text_input("CSV direct url (optional)", placeholder="paste a direct CSV url from PJM Data Miner 2 export")
                uploaded = st.file_uploader("upload 1-7 CSV files (optional)", type=["csv"], accept_multiple_files=True)
                if st.button("⚙️ auto-fill from data above"):
                    try:
                        df_list = []
                        if csv_url:
                            df_list.append(read_da_from_csv_url(csv_url))
                        if uploaded:
                            df_list.append(read_da_from_uploaded_files(uploaded))
                        if not df_list:
                            st.warning("please provide CSV link or upload files")
                        else:
                            da_df = pd.concat(df_list, axis=0, ignore_index=True).sort_values('datetime_beginning_utc').drop_duplicates('datetime_beginning_utc')
                            pred_dt = pd.Timestamp.combine(pred_date, dt.time(pred_hour))
                            feats = compute_da_features_for_prediction(da_df, pred_dt)
                            # fill 7-day lags (write to prefill_ keys)
                            for i in range(7):
                                st.session_state[f'prefill_da_price_{i}'] = round(feats[f'total_lmp_da_lag_{i+1}'], 4)
                            # fill rolling windows (write to prefill_ keys)
                            st.session_state['prefill_roll_3h_mean'] = round(feats['roll_3h_mean'], 4)
                            st.session_state['prefill_roll_6h_mean'] = round(feats['roll_6h_mean'], 4)
                            st.session_state['prefill_roll_12h_mean'] = round(feats['roll_12h_mean'], 4)
                            st.session_state['prefill_roll_24h_mean'] = round(feats['roll_24h_mean'], 4)
                            st.session_state['prefill_roll_3h_std'] = round(feats['roll_3h_std'], 4)
                            st.session_state['prefill_roll_6h_std'] = round(feats['roll_6h_std'], 4)
                            st.session_state['prefill_roll_12h_std'] = round(feats['roll_12h_std'], 4)
                            st.session_state['prefill_roll_24h_std'] = round(feats['roll_24h_std'], 4)
                            st.success("✅ automatically filled 7-day lags and rolling window features from CSV")
                            st.rerun()
                    except Exception as e:
                        st.error(f"auto-fill failed: {e}")
        
        with col2:
            st.subheader("📈 rolling window features")
            st.write("please input the price statistics features for different time windows (based on historical DA price):")
            
            col_roll1, col_roll2 = st.columns(2)
            
            with col_roll1:
                roll_3h_mean = st.number_input(
                    "3 hours mean ($/MWh)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.get('prefill_roll_3h_mean', 50.0),
                    step=1.0,
                    help="average of the past 3 hours DA price",
                    key="roll_3h_mean"
                )
                
                roll_6h_mean = st.number_input(
                    "6 hours mean ($/MWh)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.get('prefill_roll_6h_mean', 50.0),
                    step=1.0,
                    help="average of the past 6 hours DA price",
                    key="roll_6h_mean"
                )
                
                roll_12h_mean = st.number_input(
                    "12 hours mean ($/MWh)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.get('prefill_roll_12h_mean', 50.0),
                    step=1.0,
                    help="average of the past 12 hours DA price",
                    key="roll_12h_mean"
                )
                
                roll_24h_mean = st.number_input(
                    "24 hours mean ($/MWh)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=st.session_state.get('prefill_roll_24h_mean', 50.0),
                    step=1.0,
                    help="average of the past 24 hours DA price",
                    key="roll_24h_mean"
                )
            
            with col_roll2:
                roll_3h_std = st.number_input(
                    "3 hours standard deviation ($/MWh)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get('prefill_roll_3h_std', 5.0),
                    step=0.1,
                    help="standard deviation of the past 3 hours DA price",
                    key="roll_3h_std"
                )
                
                roll_6h_std = st.number_input(
                    "6 hours standard deviation ($/MWh)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get('prefill_roll_6h_std', 5.0),
                    step=0.1,
                    help="standard deviation of the past 6 hours DA price",
                    key="roll_6h_std"
                )
                
                roll_12h_std = st.number_input(
                    "12 hours standard deviation ($/MWh)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get('prefill_roll_12h_std', 5.0),
                    step=0.1,
                    help="standard deviation of the past 12 hours DA price",
                    key="roll_12h_std"
                )
                
                roll_24h_std = st.number_input(
                    "24 hours standard deviation ($/MWh)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get('prefill_roll_24h_std', 5.0),
                    step=0.1,
                    help="standard deviation of the past 24 hours DA price",
                    key="roll_24h_std"
                )
            
            st.subheader("🌤️ weather features")
            st.write("automatically get weather data for all 13 zones and aggregate")
            
            # automatically get weather forecast button
            col_weather1, col_weather2 = st.columns([1, 1])
            
            with col_weather1:
                if st.button("🌤️ get all zones weather forecast", help="get weather forecast for all 13 zones and aggregate"):
                    with st.spinner("getting all zones weather forecast..."):
                        weather_data = get_all_zones_weather_forecast(pred_date, pred_hour)
                        if weather_data:
                            # update session state
                            for key in ['agg_temp_mean', 'agg_temp_max', 'agg_temp_min', 'agg_temp_std',
                                       'agg_wind_mean', 'agg_wind_max', 'agg_pressure_mean', 
                                       'agg_soil_temp_mean', 'agg_soil_moisture_mean']:
                                if key in weather_data:
                                    st.session_state[key] = weather_data[key]
                            
                            successful_zones = weather_data.get('successful_zones', 0)
                            failed_zones = weather_data.get('failed_zones', [])
                            
                            st.success(f"✅ successfully get {successful_zones}/13 zones weather forecast")
                            if failed_zones:
                                st.warning(f"⚠️ failed to get weather forecast for the following zones: {', '.join(failed_zones)}")
                        else:
                            st.error("❌ failed to get weather forecast, using default values")
            
            # display aggregated weather features (read-only)
            st.write("**aggregated weather features:**")
            col_weather_display1, col_weather_display2 = st.columns(2)
            
            with col_weather_display1:
                temp_mean = st.number_input(
                    "average temperature (°C)",
                    value=st.session_state.get('agg_temp_mean', 20.0),
                    format="%.2f",
                    disabled=True,
                    help="average temperature of all zones"
                )
                
                temp_max = st.number_input(
                    "maximum temperature (°C)",
                    value=st.session_state.get('agg_temp_max', 25.0),
                    format="%.2f", 
                    disabled=True,
                    help="maximum temperature of all zones"
                )
                
                temp_min = st.number_input(
                    "minimum temperature (°C)",
                    value=st.session_state.get('agg_temp_min', 15.0),
                    format="%.2f",
                    disabled=True,
                    help="minimum temperature of all zones"
                )
                
                temp_std = st.number_input(
                    "temperature standard deviation (°C)",
                    value=st.session_state.get('agg_temp_std', 3.0),
                    format="%.2f",
                    disabled=True,
                    help="standard deviation of temperature of all zones"
                )
            
            with col_weather_display2:
                wind_mean = st.number_input(
                    "average wind speed (km/h)",
                    value=st.session_state.get('agg_wind_mean', 15.0),
                    format="%.2f",
                    disabled=True,
                    help="average wind speed of all zones"
                )
                
                wind_max = st.number_input(
                    "maximum wind speed (km/h)",
                    value=st.session_state.get('agg_wind_max', 20.0),
                    format="%.2f",
                    disabled=True,
                    help="maximum wind speed of all zones"
                )
                
                pressure_mean = st.number_input(
                    "average pressure (hPa)",
                    value=st.session_state.get('agg_pressure_mean', 1013.25),
                    format="%.2f",
                    disabled=True,
                    help="average pressure of all zones"
                )
                
                soil_temp_mean = st.number_input(
                    "average soil temperature (°C)",
                    value=st.session_state.get('agg_soil_temp_mean', 18.0),
                    format="%.2f",
                    disabled=True,
                    help="average soil temperature of all zones"
                )
            
            soil_moisture_mean = st.number_input(
                "average soil moisture (m³/m³)",
                value=st.session_state.get('agg_soil_moisture_mean', 0.3),
                format="%.3f",
                disabled=True,
                help="average soil moisture of all zones"
            )
        
        # prediction button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔮 make prediction", type="primary", use_container_width=True):
                # make prediction
                rolling_features = {
                    'roll_3h_mean': roll_3h_mean,
                    'roll_3h_std': roll_3h_std,
                    'roll_6h_mean': roll_6h_mean,
                    'roll_6h_std': roll_6h_std,
                    'roll_12h_mean': roll_12h_mean,
                    'roll_12h_std': roll_12h_std,
                    'roll_24h_mean': roll_24h_mean,
                    'roll_24h_std': roll_24h_std
                }
                
                weather_features = {
                    'agg_temp_mean': temp_mean,
                    'agg_temp_max': st.session_state.get('agg_temp_max', 25.0),
                    'agg_temp_min': st.session_state.get('agg_temp_min', 15.0),
                    'agg_temp_std': st.session_state.get('agg_temp_std', 3.0),
                    'agg_wind_mean': wind_mean,
                    'agg_wind_max': st.session_state.get('agg_wind_max', 20.0),
                    'agg_pressure_mean': pressure_mean,
                    'agg_soil_temp_mean': soil_temp_mean,
                    'agg_soil_moisture_mean': soil_moisture_mean
                }
                
                prediction_result = make_prediction(
                    model, scaler, feature_columns,
                    pred_date, pred_hour, da_prices,
                    rolling_features, weather_features
                )
                
                if prediction_result:
                    display_prediction_results(prediction_result)
    
    
    with tab3:
        st.header("ℹ️ usage instructions")
        
        st.markdown("""
        ### 📖 how to use this prediction system
        
        #### 1. input parameters
         - **prediction date and hour**: select the specific time point you want to predict
         - **historical DA price**: input the day-ahead market price for the past 7 days
         - **rolling window features**: input the price statistics features for different time windows
         - **weather features**: can automatically get weather forecast from API, or manually input
        
        #### 2. prediction result explanation
        - **normal**: market is running normally, the price deviation is within the expected range
        - **high anomaly**: the real-time price may significantly higher than the day-ahead price
        - **low anomaly**: the real-time price may significantly lower than the day-ahead price
        
        #### 3. risk level
        - 🟢 **low risk**: suggest normal trading according to plan
        - 🟡 **medium risk**: need to pay attention to market changes
        - 🔴 **high risk**: suggest adjusting trading strategy
        
        #### 4. notes
        - prediction result is for reference only, not investment advice
        - the quality of input data directly affects the prediction accuracy
        - it is recommended to combine other market information for comprehensive judgment
        
        #### 5. technical support
        - ensure all dependencies are installed correctly
        - if there is a problem, please check if the model file exists
        - it is recommended to retrain the model regularly to maintain accuracy
        """)

def make_prediction(model, scaler, feature_columns, pred_date, pred_hour, 
                   da_prices, rolling_features, weather_features):
    """make prediction"""
    try:
        # create prediction time
        pred_datetime = dt.datetime.combine(pred_date, dt.time(pred_hour))
        
        # create input data DataFrame
        input_data = pd.DataFrame({
            'datetime_beginning_utc': [pred_datetime],
            'hour': [pred_hour],
            'day_of_week': [pred_date.weekday()],
            'month': [pred_date.month],
            'day_of_year': [pred_date.timetuple().tm_yday],
        })
        
        # add hour one-hot encoding
        for h in range(24):
            input_data[f'hour_{h}'] = 1 if pred_hour == h else 0
        
        # add day of week one-hot encoding
        for d in range(7):
            input_data[f'dow_{d}'] = 1 if pred_date.weekday() == d else 0
        
        # add historical DA price lag features
        for i, price in enumerate(da_prices):
            input_data[f'total_lmp_da_lag_{i+1}'] = price
        
        # add real rolling window features (no longer using fake data)
        windows = [3, 6, 12, 24]
        for w in windows:
            input_data[f'da_roll_mean_{w}h'] = rolling_features[f'roll_{w}h_mean']
            input_data[f'da_roll_std_{w}h'] = rolling_features[f'roll_{w}h_std']
        
        # add aggregated weather features (from all 13 zones)
        for key, value in weather_features.items():
            input_data[key] = value
        
        # add holiday features
        input_data = apply_holiday_features(input_data, holidays)
        
        # prepare feature matrix
        X_pred = pd.DataFrame()
        for col in feature_columns:
            if col in input_data.columns:
                X_pred[col] = input_data[col]
            else:
                X_pred[col] = 0
        
        # standardize features
        X_pred_scaled = scaler.transform(X_pred)
        
        # make prediction
        prediction = model.predict(X_pred_scaled)[0]
        prediction_proba = model.predict_proba(X_pred_scaled)[0]
        
        return {
            'prediction': prediction,
            'probabilities': prediction_proba,
            'datetime': pred_datetime,
            'input_data': input_data,
            'da_prices': da_prices,
            'weather_features': weather_features,
            'rolling_features': rolling_features
        }
        
    except Exception as e:
        st.error(f"error during prediction: {e}")
        return None

def display_prediction_results(result):
    """display prediction results"""
    st.success("✅ prediction completed!")
    
    prediction = result['prediction']
    probabilities = result['probabilities']
    pred_datetime = result['datetime']
    
    # get prediction label and risk assessment
    prediction_label = get_prediction_label(prediction)
    risk_level, risk_color = get_risk_level(prediction, probabilities)
    recommendation = get_recommendation(prediction, probabilities)
    
    # display main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "prediction result",
            prediction_label,
            help="anomaly detection result based on historical data and current conditions"
        )
    
    with col2:
        st.metric(
            "risk level",
            risk_level,
            help="risk assessment based on prediction probability"
        )
    
    with col3:
        confidence = max(probabilities) * 100
        st.metric(
            "prediction confidence",
            f"{confidence:.1f}%",
            help="confidence of the model in the prediction result"
        )
    
    # display probability distribution
    st.subheader("📊 anomaly detection probability distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # create probability histogram
        categories = ['normal', 'high anomaly', 'low anomaly']
        colors = ['green', 'red', 'blue']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=probabilities,
                marker_color=colors,
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="anomaly detection probability",
            xaxis_title="category",
            yaxis_title="probability",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 detailed probability")
        for i, (category, prob) in enumerate(zip(categories, probabilities)):
            color = colors[i]
            st.write(f"**{category}**: {prob:.3f} ({prob*100:.1f}%)")
        
        st.divider()
        st.subheader("⚠️ risk assessment")
        if risk_color == 'green':
            st.success(f"🟢 {risk_level}")
        elif risk_color == 'yellow':
            st.warning(f"🟡 {risk_level}")
        else:
            st.error(f"🔴 {risk_level}")
    
    # display input summary and recommendation
    st.subheader("📋 input summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**prediction time**: {pred_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**historical DA price**: ${result['da_prices'][0]:.1f} ~ ${result['da_prices'][-1]:.1f}")
        rolling = result['rolling_features']
        st.write(f"**3 hour mean/std**: ${rolling['roll_3h_mean']:.1f} / ${rolling['roll_3h_std']:.1f}")
        st.write(f"**24 hour mean/std**: ${rolling['roll_24h_mean']:.1f} / ${rolling['roll_24h_std']:.1f}")
    
    with col2:
        weather = result['weather_features']
        st.write(f"**average temperature**: {weather['agg_temp_mean']:.1f}°C")
        st.write(f"**temperature range**: {weather['agg_temp_min']:.1f}~{weather['agg_temp_max']:.1f}°C")
        st.write(f"**average wind speed**: {weather['agg_wind_mean']:.1f} km/h")
        st.write(f"**maximum wind speed**: {weather['agg_wind_max']:.1f} km/h")
        st.write(f"**average pressure**: {weather['agg_pressure_mean']:.1f} hPa")
        is_holiday = result['input_data']['isHoliday'].iloc[0]
        st.write(f"**is holiday**: {'yes' if is_holiday else 'no'}")
        st.write(f"**prediction time**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.subheader("💡 trading recommendation")
    st.info(recommendation)

def get_prediction_label(prediction):
    """get prediction label"""
    labels = {0: "normal", 1: "high anomaly", 2: "low anomaly"}
    return labels.get(prediction, "unknown")

def get_risk_level(prediction, probabilities):
    """get risk level"""
    if prediction == 0:
        return "low risk", "green"
    elif prediction == 1:
        if probabilities[1] > 0.8:
            return "high risk", "red"
        else:
            return "medium risk", "yellow"
    else:  # prediction == 2
        if probabilities[2] > 0.8:
            return "high risk", "red"
        else:
            return "medium risk", "yellow"

def get_recommendation(prediction, probabilities):
    """get trading recommendation"""
    if prediction == 0:
        return "🟢 market is running normally, suggest normal trading according to day-ahead market price."
    elif prediction == 1:
        if probabilities[1] > 0.8:
            return "🔴 high anomaly warning! real-time price may significantly higher than day-ahead price, strongly suggest reducing real-time market purchase or increasing sale."
        else:
            return "🟡 medium risk, real-time price may higher than day-ahead price, suggest paying attention to supply and demand balance, consider适当调整交易策略。"
    else:  # prediction == 2
        if probabilities[2] > 0.8:
            return "🔴 high anomaly warning! real-time price may significantly lower than day-ahead price, strongly suggest increasing real-time market purchase or reducing sale."
        else:
            return "🟡 medium risk, real-time price may lower than day-ahead price, suggest paying attention to supply and demand balance, consider适当调整交易策略。"
        if probabilities[2] > 0.8:
            return "🔴 high anomaly warning! real-time price may significantly lower than day-ahead price, strongly suggest increasing real-time market purchase or reducing sale."
        else:
            return "🟡 medium risk, real-time price may lower than day-ahead price, suggest paying attention to supply and demand balance, consider适当调整交易策略。"

if __name__ == "__main__":
    main() 