#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electricity market anomaly detection model

Use 2022,2023,2024 data to train, 2025 data to test

Based on relative difference |RT-DA|/|RT| to detect electricity market unexpected situations
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import shap
import pickle
import joblib

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

def load_and_process_lmp_data(da_file, rt_file, weather_dir=None):
    """
    load and process LMP data
    
    Parameters:
    - da_file: path to day-ahead LMP data file
    - rt_file: path to real-time LMP data file  
    - weather_dir: path to the directory containing weather data files (optional)
    
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
    combined_data['relative_diff'] = combined_data['lmp_diff'] / combined_data['abs_rt']
    
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
    if weather_dir and os.path.isdir(weather_dir):
        print(f"add weather features from {weather_dir}...")
        
        combined_data['datetime_rounded'] = combined_data['datetime_beginning_utc'].dt.round('H')
        
        weather_cols_mapping = {
            'apparent_temperature (°C)': 'temp',
            'wind_gusts_10m (km/h)': 'wind',
            'pressure_msl (hPa)': 'pressure',
            'soil_temperature_0_to_7cm (°C)': 'soil_temp',
            'soil_moisture_0_to_7cm (m³/m³)': 'soil_moisture'
        }
        
        all_weather_dfs = []
        for weather_file in os.listdir(weather_dir):
            if weather_file.endswith(".pkl"):
                zone_name = weather_file.split('_')[0]
                weather_file_path = os.path.join(weather_dir, weather_file)
                try:
                    weather_data = pd.read_pickle(weather_file_path)
                    if 'time' not in weather_data.columns:
                        weather_data = weather_data.reset_index().rename(columns={'index': 'time'})
                    weather_data['time'] = pd.to_datetime(weather_data['time'])
                    weather_data['datetime_rounded'] = weather_data['time'].dt.round('H')
                    
                    # Keep original weather column names for now
                    weather_data = weather_data[['datetime_rounded'] + list(weather_cols_mapping.keys())]
                    
                    # Prefix columns
                    rename_dict = {col: f"{zone_name}_{col}" for col in weather_cols_mapping.keys()}
                    weather_data.rename(columns=rename_dict, inplace=True)
                    
                    all_weather_dfs.append(weather_data.set_index('datetime_rounded'))
                    
                except Exception as e:
                    print(f"⚠️ weather data processing failed for {weather_file}: {e}")

        # Merge all weather dataframes at once
        if all_weather_dfs:
            # Using outer join to keep all timestamps
            merged_weather = pd.concat(all_weather_dfs, axis=1, join='outer')
            # Now merge with combined_data
            combined_data = combined_data.set_index('datetime_rounded').join(merged_weather).reset_index()

        print(f"✅ weather features added")

        # --- Create aggregated weather features ---
        print("Creating aggregated weather features...")
        weather_feature_cols = {}
        for original_name, short_name in weather_cols_mapping.items():
            weather_feature_cols[short_name] = [col for col in combined_data.columns if original_name in col]

        # Temperature aggregates
        temp_cols = weather_feature_cols['temp']
        if temp_cols:
            combined_data['agg_temp_mean'] = combined_data[temp_cols].mean(axis=1)
            combined_data['agg_temp_max'] = combined_data[temp_cols].max(axis=1)
            combined_data['agg_temp_min'] = combined_data[temp_cols].min(axis=1)
            combined_data['agg_temp_std'] = combined_data[temp_cols].std(axis=1)

        # Wind aggregates
        wind_cols = weather_feature_cols['wind']
        if wind_cols:
            combined_data['agg_wind_mean'] = combined_data[wind_cols].mean(axis=1)
            combined_data['agg_wind_max'] = combined_data[wind_cols].max(axis=1)
            
        # Other aggregates (mean)
        for short_name in ['pressure', 'soil_temp', 'soil_moisture']:
            cols = weather_feature_cols[short_name]
            if cols:
                combined_data[f'agg_{short_name}_mean'] = combined_data[cols].mean(axis=1)
        
        # Drop original weather columns
        all_original_weather_cols = [col for sublist in weather_feature_cols.values() for col in sublist]
        combined_data.drop(columns=all_original_weather_cols, inplace=True, errors='ignore')
        print("✅ Aggregated features created and original weather columns dropped.")


    # Fill any missing weather data (from merge or if file failed) with defaults
    agg_weather_cols = [col for col in combined_data.columns if col.startswith('agg_')]
    for col in agg_weather_cols:
        combined_data[col] = combined_data[col].fillna(combined_data[col].mean())
    
    # add holiday features
    combined_data = apply_holiday_features(combined_data, holidays)
    
    # delete rows with NaN values (due to lag features)
    # The first 7 days will have NaNs due to the 7-day lag features.
    # We should only drop rows that have NaNs in the lag columns.
    lag_cols = [f'total_lmp_da_lag_{lag}' for lag in range(1, 8)]
    combined_data = combined_data.dropna(subset=lag_cols).reset_index(drop=True)
    
    print(f"✅ data processing completed: {len(combined_data)} rows, {len(combined_data.columns)} columns")
    
    return combined_data

def create_anomaly_target(data, method='statistical', k=2.0, percentile=95):
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
        merged_data['deviation'] = merged_data['relative_diff'] - merged_data['mu']
        merged_data['threshold'] = k * merged_data['sigma']
        merged_data['sum']=merged_data['mu'] + merged_data['threshold']
        merged_data['difference']=merged_data['mu']- merged_data['threshold']
        merged_data[['mu', 'threshold','relative_diff','sum','difference']].to_csv(f'mu_threshold_k{k}.csv', index=False)  # save to csv with k value
        merged_data['target'] = 0
        merged_data.loc[merged_data['deviation'] > merged_data['threshold'], 'target'] = 1  # high anomaly
        merged_data.loc[merged_data['deviation'] < -merged_data['threshold'], 'target'] = 2  # low anomaly
        
        data['target'] = merged_data['target']
        
    else:  # percentile method
        print(f"=== use percentile method to define anomaly ({percentile}th percentile) ===")
        
        threshold_high = np.percentile(data['relative_diff'], percentile)
        threshold_low = np.percentile(data['relative_diff'], 100 - percentile)
        data['target'] = 0
        data.loc[data['relative_diff'] >= threshold_high, 'target'] = 1
        data.loc[data['relative_diff'] <= threshold_low, 'target'] = 2
    
    # statistical results
    anomaly_count = (data['target'] != 0).sum()
    high_count = (data['target'] == 1).sum()
    low_count = (data['target'] == 2).sum()
    anomaly_rate = anomaly_count / len(data) * 100
    
    print(f"anomaly detection results:")
    print(f"total anomalies: {anomaly_count} (high: {high_count}, low: {low_count})")
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
        
        # time features
        'hour', 'day_of_week', 'month', 'day_of_year',
        
        # holiday features
        'isHoliday'
    ]
    
    # Dynamically add aggregated weather features
    agg_weather_cols = [col for col in data.columns if col.startswith('agg_')]
    feature_columns.extend(agg_weather_cols)
    
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
    """use 2022,2023,2024 data to train model"""
    print("=== use 2022,2023,2024 data to train model ===")
    
    # define multiple years data files
    data_files = [
        {
            "da_file": "applicationData/da_hrl_lmps_2022.csv",
            "rt_file": "applicationData/rt_hrl_lmps_2022.csv",
            "weather_dir": "weatherData/meteo/"
        },
        {
            "da_file": "applicationData/da_hrl_lmps_2023.csv",
            "rt_file": "applicationData/rt_hrl_lmps_2023.csv",
            "weather_dir": "weatherData/meteo/"
        },
        {
            "da_file": "applicationData/da_hrl_lmps_2024.csv",
            "rt_file": "applicationData/rt_hrl_lmps_2024.csv",
            "weather_dir": "weatherData/meteo/"
        }
    ]
    
    # load and merge all data
    all_train_data = []
    for files in data_files:
        print(f"\nLoading data for {files['da_file']}...")
        data = load_and_process_lmp_data(
            da_file=files["da_file"],
            rt_file=files["rt_file"],
            weather_dir=files["weather_dir"]
        )
        if data is not None:
            all_train_data.append(data)
    
    if not all_train_data:
        print("❌ No training data loaded. Exiting.")
        return None, None, None, None

    train_data = pd.concat(all_train_data, ignore_index=True)
    print(f"\nTotal training data rows after combining all years: {len(train_data)}")
    
    # create anomaly target variable
    train_data = create_anomaly_target(train_data, method='statistical', k=2.0)
    
    # prepare features
    feature_columns, X_train, y_train = prepare_features_for_prediction(train_data)
    
    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use full training data for tuning and training
    X_train_split, y_train_split = X_train_scaled, y_train

    print(f"training set feature shape: {X_train_split.shape}")
    print(f"training set target distribution: {pd.Series(y_train_split).value_counts().to_dict()}")

    # --- RandomForest Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Tuning RandomForest hyperparameters with GridSearchCV ---")
    
    # Define the parameter grid to search
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    # Create a GridSearchCV object
    rf = RandomForestClassifier(random_state=42)
    # Using scoring='f1_macro' for multi-class
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                                  scoring='f1_macro', cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the training set
    grid_search_rf.fit(X_train_split, y_train_split)

    print("\nBest parameters found for RandomForest:")
    print(grid_search_rf.best_params_)

    # Get the best model from the grid search
    best_rf_model = grid_search_rf.best_estimator_

    # train multiple models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            multi_class='multinomial',
            max_iter=2000,
            random_state=42
        )
    }
    
    # Start the dictionary of trained models with the optimized RandomForest
    trained_models = {'RandomForest': best_rf_model}
    
    for name, model in models.items():
        print(f"\ntrain {name}...")
        # Train on the training data
        model.fit(X_train_split, y_train_split)
        
        # calculate training set performance
        train_score = model.score(X_train_split, y_train_split)
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
        weather_dir="weatherData/meteo/"
    )
    
    # create anomaly target variable (using same method)
    test_data = create_anomaly_target(test_data, method='statistical', k=2.0)
    
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
        
        y_pred = model.predict(X_test_scaled)

        # calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
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
            'predictions': y_pred
        }
    
    return results, test_data

def find_optimal_threshold(y_true, y_pred_proba, model_name=""):
    """
    Find the optimal prediction threshold for a model
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # F1 score for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    # locate the index of the largest f1 score
    f1_scores = np.nan_to_num(f1_scores) # handle division by zero
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    print(f"  Best Threshold={best_threshold:.4f}, F1-Score={f1_scores[best_f1_idx]:.4f}")

    # plot the roc curve for the model
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score')
    plt.axvline(x=best_threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({best_threshold:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, and F1 Score vs. Threshold for {model_name}')
    plt.legend()
    plt.show()

    return best_threshold


def analyze_prediction_patterns(test_data, results):
    """analyze prediction patterns and anomalies"""
    print("\n=== analyze prediction patterns and anomalies ===")
    
    # analyze actual anomalies
    anomalies = test_data[test_data['target'] != 0]
    
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
    print(f"  min: {test_data['relative_diff'].min():.4f}")
    if len(anomalies) > 0:
        print(f"  average relative difference of anomalies: {anomalies['relative_diff'].mean():.4f}")
    
    # find the largest anomalies
    print(f"\ntop 10 largest anomalies (high):")
    top_high = test_data.nlargest(10, 'relative_diff')
    for idx, row in top_high.iterrows():
        date_str = row['datetime_beginning_utc'].strftime('%Y-%m-%d %H:%M')
        print(f"  {date_str}: relative difference={row['relative_diff']:.4f}, DA=${row['total_lmp_da']:.2f}, RT=${row['total_lmp_rt']:.2f}")
    
    print(f"\ntop 10 smallest anomalies (low):")
    top_low = test_data.nsmallest(10, 'relative_diff')
    for idx, row in top_low.iterrows():
        date_str = row['datetime_beginning_utc'].strftime('%Y-%m-%d %H:%M')
        print(f"  {date_str}: relative difference={row['relative_diff']:.4f}, DA=${row['total_lmp_da']:.2f}, RT=${row['total_lmp_rt']:.2f}")
        
def plot_feature_importance(model, feature_names, top_n=30, model_name=""):
    """
    Plots the feature importance of a trained model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])  # For binary classification
    else:
        print(f"Model {model_name} does not support feature importance.")
        return
    
    # Use numpy array for easier indexing
    feature_names = np.array(feature_names)
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(12, 10))
    plt.title(f'Top {top_n} Feature Importances for {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='maroon', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

    # Print the top N features
    print(f"\nTop {top_n} most important features for {model_name}:")
    for i in np.argsort(importances)[::-1][:top_n]:
        print(f"- {feature_names[i]}: {importances[i]:.4f}")


def test_different_k_values():
    """test different k values"""
    print("=== test different k values ===")
    
    # k values to test
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    # 1. train model (using default k=2.0)
    print("\n=== 训练模型 ===")
    trained_models, scaler, feature_columns, train_data = train_models_2024_data()
    
    # 2. load test data
    print("\n=== load 2025 test data ===")
    test_data = load_and_process_lmp_data(
        da_file="applicationData/da_hrl_lmps_2025.csv",
        rt_file="applicationData/rt_hrl_lmps_2025.csv",
        weather_dir="weatherData/meteo/"
    )
    
    # 3. test different k values
    k_results = {}
    
    for k in k_values:
        print(f"\n{'='*20} 测试 k={k} {'='*20}")
        
        # create anomaly target variable using current k value
        test_data_k = create_anomaly_target(test_data.copy(), method='statistical', k=k)
        
        # prepare features (ensure feature order is consistent)
        X_test = pd.DataFrame()
        for col in feature_columns:
            if col in test_data_k.columns:
                X_test[col] = test_data_k[col]
            else:
                X_test[col] = 0  # fill missing features with 0
        
        y_test = test_data_k['target']
        
        # standardize features
        X_test_scaled = scaler.transform(X_test)
        
        print(f"test set feature shape: {X_test_scaled.shape}")
        print(f"test set target distribution: {y_test.value_counts().to_dict()}")
        
        # test each model
        k_model_results = {}
        
        for name, model in trained_models.items():
            print(f"\ntest {name}...")
            
            y_pred = model.predict(X_test_scaled)

            # calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            print(f"{name} test results:")
            print(f"  accuracy: {accuracy:.3f}")
            print(f"  precision: {precision:.3f}")
            print(f"  recall: {recall:.3f}")
            print(f"  F1 score: {f1:.3f}")
            
            # confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"  confusion matrix:\n{cm}")
            
            k_model_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'anomaly_count': (y_test != 0).sum(),
                'anomaly_rate': (y_test != 0).sum() / len(y_test) * 100
            }
        
        k_results[k] = k_model_results
    
    # 4. compare different k values
    print(f"\n{'='*50}")
    print("📊 different k values results")
    print(f"{'='*50}")
    
    # create comparison table
    print(f"\n{'k value':<8} {'anomaly rate(%)':<12} {'XGBoost F1':<12} {'RandomForest F1':<15} {'LogisticRegression F1':<20}")
    print("-" * 70)
    
    for k in k_values:
        anomaly_rate = k_results[k]['XGBoost']['anomaly_rate']
        xgb_f1 = k_results[k]['XGBoost']['f1']
        rf_f1 = k_results[k]['RandomForest']['f1']
        lr_f1 = k_results[k]['LogisticRegression']['f1']
        
        print(f"{k:<8.1f} {anomaly_rate:<12.2f} {xgb_f1:<12.3f} {rf_f1:<15.3f} {lr_f1:<20.3f}")
    
    # visualize different k values
    plt.figure(figsize=(15, 10))
    
    # subplot 1: F1 score comparison
    plt.subplot(2, 2, 1)
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        f1_scores = [k_results[k][model_name]['f1'] for k in k_values]
        plt.plot(k_values, f1_scores, marker='o', label=model_name)
    plt.xlabel('k value')
    plt.ylabel('F1 score')
    plt.title('F1 score vs. k value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # subplot 2: anomaly rate change
    plt.subplot(2, 2, 2)
    anomaly_rates = [k_results[k]['XGBoost']['anomaly_rate'] for k in k_values]
    plt.plot(k_values, anomaly_rates, marker='s', color='red', linewidth=2)
    plt.xlabel('k value')
    plt.ylabel('anomaly rate (%)')
    plt.title('anomaly rate vs. k value')
    plt.grid(True, alpha=0.3)
    
    # subplot 3: accuracy comparison
    plt.subplot(2, 2, 3)
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        accuracies = [k_results[k][model_name]['accuracy'] for k in k_values]
        plt.plot(k_values, accuracies, marker='^', label=model_name)
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. k value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # subplot 4: precision comparison
    plt.subplot(2, 2, 4)
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        precisions = [k_results[k][model_name]['precision'] for k in k_values]
        plt.plot(k_values, precisions, marker='d', label=model_name)
    plt.xlabel('k value')
    plt.ylabel('precision')
    plt.title('precision vs. k value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_values_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # find the best k value
    print(f"\n🏆 best k value recommended:")
    best_k_by_model = {}
    
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        f1_scores = {k: k_results[k][model_name]['f1'] for k in k_values}
        best_k = max(f1_scores.keys(), key=lambda k: f1_scores[k])
        best_f1 = f1_scores[best_k]
        best_k_by_model[model_name] = (best_k, best_f1)
        print(f"  {model_name}: k={best_k} (F1={best_f1:.3f})")
    
    # overall best k value (based on average F1 score)
    avg_f1_by_k = {}
    for k in k_values:
        avg_f1 = np.mean([k_results[k][model]['f1'] for model in ['XGBoost', 'RandomForest', 'LogisticRegression']])
        avg_f1_by_k[k] = avg_f1
    
    overall_best_k = max(avg_f1_by_k.keys(), key=lambda k: avg_f1_by_k[k])
    print(f"  overall best: k={overall_best_k} (average F1={avg_f1_by_k[overall_best_k]:.3f})")
    
    return k_results, overall_best_k

def save_model_for_prediction(trained_models, scaler, feature_columns, k_value=2.0):
    """Save model and components for prediction"""
    print("💾 save model and components...")
    
    # create model directory
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # save XGBoost model (main used model)
    xgb_model = trained_models['XGBoost']
    joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))
    
    # save feature scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # save feature list
    with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # save k value
    with open(os.path.join(model_dir, 'k_value.pkl'), 'wb') as f:
        pickle.dump(k_value, f)
    
    # save model configuration information
    model_info = {
        'model_type': 'XGBoost',
        'k_value': k_value,
        'feature_count': len(feature_columns),
        'save_date': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✅ model saved to {model_dir} directory")
    print(f"   - XGBoost model: xgboost_model.pkl")
    print(f"   - feature scaler: scaler.pkl") 
    print(f"   - feature list: feature_columns.pkl")
    print(f"   - k value: k_value.pkl")
    print(f"   - model information: model_info.pkl")

def load_model_for_prediction():
    """Load saved model and components for prediction"""
    model_dir = "saved_models"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"model directory {model_dir} does not exist, please train and save model first")
    
    # load model and components
    xgb_model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    
    with open(os.path.join(model_dir, 'k_value.pkl'), 'rb') as f:
        k_value = pickle.load(f)
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
        model_info = pickle.load(f)
    
    print(f"✅ model loaded:")
    print(f"   - model type: {model_info['model_type']}")
    print(f"   - k value: {model_info['k_value']}")
    print(f"   - feature count: {model_info['feature_count']}")
    print(f"   - save date: {model_info['save_date']}")
    
    return xgb_model, scaler, feature_columns, k_value

def main():
    """main function"""
    try:
        print("🚀 start power market anomaly detection model training and testing")
        
        # train model
        print("\n=== train model ===")
        trained_models, scaler, feature_columns, train_data = train_models_2024_data()
        
        # save best model
        save_model_for_prediction(trained_models, scaler, feature_columns, k_value=3.5)
        
        # test different k values  
        k_results, best_k = test_different_k_values()
        
        print(f"\n✅ different k values test completed! recommend using k={best_k}")
        print(f"📊 results visualization saved as k_values_comparison.png")
        
    except Exception as e:
        print(f"❌ training and testing process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 