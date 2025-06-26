#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electricity market anomaly detection model

Use 2024 data to train, 2025 data to test

Based on relative difference |RT-DA|/|RT| to detect electricity market unexpected situations
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# holiday definition
holidays = [
    {'holiday_name': "New Year's Day", 'month': 1, 'date': 1, 'occurence': 1, 'startYear': 1870},
    {'holiday_name': "Independence Day", 'month': 7, 'date': 4, 'occurence': 1, 'startYear': 1870},
    {'holiday_name': "Christmas Day", 'month': 12, 'date': 25, 'occurence': 1, 'startYear': 1870},
    {'holiday_name': "Veterans Day", 'month': 11, 'date': 11, 'occurence': 1, 'startYear': 1954},
    {'holiday_name': "Martin Luther King Jr", 'month': 1, 'date': "monday", 'occurence': 3, 'startYear': 1986}
]

def apply_holiday_features(data, holiday_list):
    """add holiday features (efficient version)"""
    data = data.copy()
    
    # create a set of holiday dates
    holiday_dates = set()
    years = data['datetime_beginning_utc'].dt.year.unique()

    for item in holiday_list:
        # only process fixed date holidays
        if isinstance(item['date'], int):
            for year in years:
                if year >= item['startYear']:
                    try:
                        holiday_dates.add(pd.to_datetime(f"{year}-{item['month']}-{item['date']}").date())
                    except ValueError:
                        # Handles cases like leap years if date is invalid
                        continue
    
    # MLK Day: 3rd Monday in January
    for year in years:
        if year >= 1986:
            # Get all Mondays in January
            first_day = dt.date(year, 1, 1)
            # Find the first Monday
            first_monday = first_day + dt.timedelta(days=(7 - first_day.weekday()) % 7)
            # Third Monday
            mlk_day = first_monday + dt.timedelta(weeks=2)
            holiday_dates.add(mlk_day)

    data['isHoliday'] = data['datetime_beginning_utc'].dt.date.isin(holiday_dates).astype(int)
    return data

def load_and_process_lmp_data(da_file, rt_file, weather_file=None):
    """
    load and process LMP data
    
    Parameters:
    - da_file: path to day-ahead LMP data file
    - rt_file: path to real-time LMP data file  
    - weather_file: path to weather data file (optional)
    
    Returns:
    - processed_data: processed data DataFrame
    """
    print(f"=== load LMP data ===")
    
    # read DA and RT data
    da_data = pd.read_csv(da_file)
    rt_data = pd.read_csv(rt_file)
    
    print(f"DA data: {len(da_data)} rows")
    print(f"RT data: {len(rt_data)} rows")
    
    # ensure data length is consistent
    min_len = min(len(da_data), len(rt_data))
    da_data = da_data[:min_len].reset_index(drop=True)
    rt_data = rt_data[:min_len].reset_index(drop=True)
    
    # merge DA and RT data
    combined_data = da_data.copy()
    combined_data['total_lmp_rt'] = rt_data['total_lmp_rt']
    
    # parse time and add time features
    combined_data['datetime_beginning_utc'] = pd.to_datetime(combined_data['datetime_beginning_utc'])
    combined_data['hour'] = combined_data['datetime_beginning_utc'].dt.hour
    combined_data['day_of_week'] = combined_data['datetime_beginning_utc'].dt.dayofweek
    combined_data['month'] = combined_data['datetime_beginning_utc'].dt.month
    combined_data['day_of_year'] = combined_data['datetime_beginning_utc'].dt.dayofyear
    
    # add hour one-hot encoding
    for h in range(24):
        combined_data[f'hour_{h}'] = (combined_data['hour'] == h).astype(int)
    
    # add day of week one-hot encoding  
    for d in range(7):
        combined_data[f'dow_{d}'] = (combined_data['day_of_week'] == d).astype(int)
    
    # calculate relative difference |RT-DA|/|RT| (avoid division by zero)
    combined_data['lmp_diff'] = combined_data['total_lmp_rt'] - combined_data['total_lmp_da']
    combined_data['abs_rt'] = np.abs(combined_data['total_lmp_rt'])
    combined_data['abs_rt'] = np.where(combined_data['abs_rt'] < 1e-6, 1e-6, combined_data['abs_rt'])
    combined_data['relative_diff'] = np.abs(combined_data['lmp_diff']) / combined_data['abs_rt']
    
    # add historical price lag features (1-7 days ago DA price)
    for lag in range(1, 8):
        combined_data[f'total_lmp_da_lag_{lag}'] = combined_data['total_lmp_da'].shift(lag * 24)  # 24 hours ago

    # --- add new feature engineering ---
    print("add rolling window features...")
    windows = [3, 6, 12, 24]
    for w in windows:
        # use shift(1) to ensure window only contains past data, avoid data leakage
        shifted_data = combined_data['total_lmp_da'].shift(1)
        combined_data[f'da_roll_mean_{w}h'] = shifted_data.rolling(window=w).mean()
        combined_data[f'da_roll_std_{w}h'] = shifted_data.rolling(window=w).std()
    # --- end new feature engineering ---
    
    # add weather features (if weather data is provided)
    if weather_file and os.path.exists(weather_file):
        try:
            print(f"add weather features from {weather_file}...")
            weather_data = pd.read_pickle(weather_file)

            if 'time' not in weather_data.columns:
                weather_data = weather_data.reset_index().rename(columns={'index': 'time'})

            weather_data['time'] = pd.to_datetime(weather_data['time'])
            weather_data['datetime_rounded'] = weather_data['time'].dt.round('H')
            
            weather_cols = [
                'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
                'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)'
            ]
            
            # Select and deduplicate weather data at hourly resolution
            weather_hourly = weather_data[['datetime_rounded'] + weather_cols].drop_duplicates(subset=['datetime_rounded'], keep='first')

            # Round lmp data time to merge
            combined_data['datetime_rounded'] = combined_data['datetime_beginning_utc'].dt.round('H')
            
            combined_data = pd.merge(combined_data, weather_hourly, on='datetime_rounded', how='left')
            combined_data = combined_data.drop(columns=['datetime_rounded'])
            
            print(f"✅ weather features added")
        except Exception as e:
            print(f"⚠️ weather data processing failed: {e}")
            # fallback to default values if processing fails
            for col in weather_cols:
                combined_data[col] = 15.0 # simplified default
    
    # Fill any missing weather data (from merge or if file failed) with defaults
    default_weather = {
        'apparent_temperature (°C)': 15.0, 'wind_gusts_10m (km/h)': 20.0,
        'pressure_msl (hPa)': 1013.25, 'soil_temperature_0_to_7cm (°C)': 12.0,
        'soil_moisture_0_to_7cm (m³/m³)': 0.3
    }
    for col, val in default_weather.items():
        if col in combined_data.columns:
            combined_data[col] = combined_data[col].fillna(val)
        else:
            combined_data[col] = val
    
    # add holiday features
    combined_data = apply_holiday_features(combined_data, holidays)
    
    # delete rows with NaN values (due to lag features)
    # The first 7 days will have NaNs due to the 7-day lag features.
    # We should only drop rows that have NaNs in the lag columns.
    lag_cols = [f'total_lmp_da_lag_{lag}' for lag in range(1, 8)]
    combined_data = combined_data.dropna(subset=lag_cols).reset_index(drop=True)
    
    print(f"✅ data processing completed: {len(combined_data)} rows, {len(combined_data.columns)} columns")
    
    return combined_data

def create_anomaly_target(data, method='statistical', k=2.5, percentile=95):
    """
    create anomaly detection target variable
    
    Parameters:
    - data: data DataFrame
    - method: 'statistical' or 'percentile'
    - k: standard deviation multiplier threshold for statistical method
    - percentile: percentile threshold for percentile method
    
    Returns:
    - data: data with target variable
    """
    data = data.copy()
    
    if method == 'statistical':
        print(f"=== use statistical method to define anomaly (k={k}) ===")
        
        # calculate statistical baseline based on time period
        baseline_stats = data.groupby(['hour', 'day_of_week'])['relative_diff'].agg(
            mu='mean',
            sigma='std',
            count='count'
        ).reset_index()
        
        # handle cases where standard deviation is NaN or 0
        global_sigma = data['relative_diff'].std()
        if pd.isna(global_sigma) or global_sigma == 0:
            global_sigma = 1.0
        
        baseline_stats['sigma'] = baseline_stats['sigma'].fillna(global_sigma)
        baseline_stats.loc[baseline_stats['sigma'] == 0, 'sigma'] = global_sigma
        
        # merge baseline statistics
        merged_data = data.merge(baseline_stats, on=['hour', 'day_of_week'], how='left')
        
        # fill missing baseline statistics
        global_mu = merged_data['relative_diff'].mean()
        merged_data['mu'] = merged_data['mu'].fillna(global_mu)
        merged_data['sigma'] = merged_data['sigma'].fillna(global_sigma)
        
        # calculate anomaly
        merged_data['abs_deviation'] = np.abs(merged_data['relative_diff'] - merged_data['mu'])
        merged_data['threshold'] = k * merged_data['sigma']
        merged_data['is_anomaly'] = (merged_data['abs_deviation'] > merged_data['threshold']).astype(int)
        
        data['target'] = merged_data['is_anomaly']
        
    else:  # percentile method
        print(f"=== use percentile method to define anomaly ({percentile}th percentile) ===")
        
        threshold = np.percentile(data['relative_diff'], percentile)
        data['target'] = (data['relative_diff'] >= threshold).astype(int)
    
    # statistical results
    anomaly_count = data['target'].sum()
    anomaly_rate = anomaly_count / len(data) * 100
    
    print(f"anomaly detection results:")
    print(f"total anomalies: {anomaly_count}")
    print(f"anomaly rate: {anomaly_rate:.2f}%")
    print(f"target variable distribution: {data['target'].value_counts().to_dict()}")
    
    return data

def prepare_features_for_prediction(data):
    """
    prepare features for prediction
    """
    # select feature columns
    feature_columns = [
        # historical DA price features
        'total_lmp_da_lag_1', 'total_lmp_da_lag_2', 'total_lmp_da_lag_3',
        'total_lmp_da_lag_4', 'total_lmp_da_lag_5', 'total_lmp_da_lag_6', 'total_lmp_da_lag_7',
        
        # new rolling window features
        'da_roll_mean_3h', 'da_roll_std_3h',
        'da_roll_mean_6h', 'da_roll_std_6h',
        'da_roll_mean_12h', 'da_roll_std_12h',
        'da_roll_mean_24h', 'da_roll_std_24h',
        
        # weather features
        'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
        'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
        
        # time features
        'hour', 'day_of_week', 'month', 'day_of_year',
        
        # holiday features
        'isHoliday'
    ]
    
    # add hour one-hot encoding
    hour_cols = [f'hour_{h}' for h in range(24)]
    feature_columns.extend(hour_cols)
    
    # add day of week one-hot encoding
    dow_cols = [f'dow_{d}' for d in range(7)]
    feature_columns.extend(dow_cols)
    
    # check if features exist
    available_features = [col for col in feature_columns if col in data.columns]
    missing_features = [col for col in feature_columns if col not in data.columns]
    
    if missing_features:
        print(f"⚠️ missing features: {missing_features}")
    
    print(f"available features: {len(available_features)} / {len(feature_columns)}")
    
    X = data[available_features]
    y = data['target'] if 'target' in data.columns else None
    
    return available_features, X, y

def train_models_2024_data():
    """use 2024 data to train model"""
    print("=== use 2024 data to train model ===")
    
    # load 2024 data
    train_data = load_and_process_lmp_data(
        da_file="applicationData/da_hrl_lmps_2024.csv",
        rt_file="applicationData/rt_hrl_lmps_2024.csv",
        weather_file="weatherData/meteo/western_hub_weather_2024-01-01_to_2024-12-31.pkl"
    )
    
    # create anomaly target variable
    train_data = create_anomaly_target(train_data, method='statistical', k=2.5)
    
    # prepare features
    feature_columns, X_train, y_train = prepare_features_for_prediction(train_data)
    
    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"training set feature shape: {X_train_scaled.shape}")
    print(f"training set target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    # train multiple models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=25, # Best value from previous experiment
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\ntrain {name}...")
        model.fit(X_train_scaled, y_train)
        
        # calculate training set performance
        train_score = model.score(X_train_scaled, y_train)
        print(f"{name} training set accuracy: {train_score:.3f}")
        
        trained_models[name] = model
    
    return trained_models, scaler, feature_columns, train_data

def test_models_2025_data(trained_models, scaler, feature_columns):
    """use 2025 data to test model"""
    print("\n=== use 2025 data to test model ===")
    
    # load 2025 data
    test_data = load_and_process_lmp_data(
        da_file="applicationData/da_hrl_lmps_2025.csv",
        rt_file="applicationData/rt_hrl_lmps_2025.csv",
        weather_file="weatherData/meteo/western_hub_weather_2025-01-01_to_2025-06-06.pkl"
    )
    
    # create anomaly target variable (using same method)
    test_data = create_anomaly_target(test_data, method='statistical', k=2.5)
    
    # prepare features (ensure feature order is consistent)
    X_test = pd.DataFrame()
    for col in feature_columns:
        if col in test_data.columns:
            X_test[col] = test_data[col]
        else:
            X_test[col] = 0  # fill missing features with 0
    
    y_test = test_data['target']
    
    # standardize features
    X_test_scaled = scaler.transform(X_test)
    
    print(f"test set feature shape: {X_test_scaled.shape}")
    print(f"test set target distribution: {y_test.value_counts().to_dict()}")
    
    # test each model
    results = {}
    
    for name, model in trained_models.items():
        print(f"\ntest {name}...")
        
        # predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"{name} test results:")
        print(f"  accuracy: {accuracy:.3f}")
        print(f"  precision: {precision:.3f}")
        print(f"  recall: {recall:.3f}")
        print(f"  F1 score: {f1:.3f}")
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  confusion matrix:\n{cm}")
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return results, test_data

def analyze_prediction_patterns(test_data, results):
    """analyze prediction patterns and anomalies"""
    print("\n=== analyze prediction patterns and anomalies ===")
    
    # analyze actual anomalies
    anomalies = test_data[test_data['target'] == 1]
    
    print(f"actual anomalies: {len(anomalies)}")
    if len(anomalies) > 0:
        print(f"anomaly hour distribution:")
        hour_dist = anomalies['hour'].value_counts().sort_index()
        for hour, count in hour_dist.items():
            print(f"  {hour}: {count} times")
        
        print(f"anomaly day of week distribution:")
        dow_dist = anomalies['day_of_week'].value_counts().sort_index()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for dow, count in dow_dist.items():
            print(f"  {dow_names[dow]}: {count} times")
    
    # analyze distribution of relative difference
    print(f"\nrelative difference statistics:")
    print(f"  mean: {test_data['relative_diff'].mean():.4f}")
    print(f"  std: {test_data['relative_diff'].std():.4f}")
    print(f"  max: {test_data['relative_diff'].max():.4f}")
    if len(anomalies) > 0:
        print(f"  average relative difference of anomalies: {anomalies['relative_diff'].mean():.4f}")
    
    # find the largest anomalies
    top_anomalies = test_data.nlargest(10, 'relative_diff')
    print(f"\ntop 10 largest anomalies:")
    for idx, row in top_anomalies.iterrows():
        date_str = row['datetime_beginning_utc'].strftime('%Y-%m-%d %H:%M')
        print(f"  {date_str}: relative difference={row['relative_diff']:.4f}, DA=${row['total_lmp_da']:.2f}, RT=${row['total_lmp_rt']:.2f}")

def main():
    """main function"""
    try:
        print("🚀 start electricity market anomaly detection model training and testing")
        
        # 1. train model
        trained_models, scaler, feature_columns, train_data = train_models_2024_data()
        
        # 2. test model  
        test_results, test_data = test_models_2025_data(trained_models, scaler, feature_columns)
        
        # 3. analyze results
        analyze_prediction_patterns(test_data, test_results)
        
        # 4. save best model
        best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['f1'])
        print(f"\n🏆 best model: {best_model_name} (F1={test_results[best_model_name]['f1']:.3f})")
        
        print("\n✅ model training and testing completed!")
        
    except Exception as e:
        print(f"❌ training and testing process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 