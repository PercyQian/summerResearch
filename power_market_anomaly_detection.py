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
from sklearn.tree import DecisionTreeClassifier

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
    orig_len_da = len(da_data)
    orig_len_rt = len(rt_data)
    min_len = min(orig_len_da, orig_len_rt)
    da_truncated = orig_len_da - min_len
    rt_truncated = orig_len_rt - min_len
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
    
    # calculate relative difference |RT-DA|/|DA| (avoid division by zero)
    combined_data['lmp_diff'] = combined_data['total_lmp_rt'] - combined_data['total_lmp_da']
    combined_data['abs_da'] = np.abs(combined_data['total_lmp_da'])
    combined_data['abs_da'] = np.where(combined_data['abs_da'] < 1e-6, 1e-6, combined_data['abs_da'])
    combined_data['relative_diff'] = np.abs(combined_data['lmp_diff']) / combined_data['abs_da']
    # log smoothing for non-negative relative diff: ln(1 + x)
    combined_data['relative_diff_log'] = np.log1p(combined_data['relative_diff'])
    
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

    # === diagnostics: which rows will be dropped due to missing lag features ===
    try:
        # build row index to distinguish warm-up rows
        combined_data['__row_idx__'] = np.arange(len(combined_data))

        lag_cols = [f'total_lmp_da_lag_{lag}' for lag in range(1, 8)]
        lag_na = {k: combined_data[f'total_lmp_da_lag_{k}'].isna() for k in range(1, 8)}
        warmup = {k: (combined_data['__row_idx__'] < (24 * k)) for k in range(1, 8)}
        missing_source = {k: (lag_na[k] & ~warmup[k]) for k in range(1, 8)}

        # any lag missing for this row
        any_lag_missing = None
        for k in range(1, 8):
            any_lag_missing = lag_na[k] if any_lag_missing is None else (any_lag_missing | lag_na[k])

        drop_candidates = combined_data[any_lag_missing].copy()

        # annotate reasons
        if not drop_candidates.empty:
            for k in range(1, 8):
                drop_candidates[f'warmup_lag_{k}'] = warmup[k][any_lag_missing].values
                drop_candidates[f'missing_source_lag_{k}'] = missing_source[k][any_lag_missing].values

            drop_candidates['any_warmup'] = False
            drop_candidates['any_missing_source'] = False
            for k in range(1, 8):
                drop_candidates['any_warmup'] = drop_candidates['any_warmup'] | drop_candidates[f'warmup_lag_{k}']
                drop_candidates['any_missing_source'] = drop_candidates['any_missing_source'] | drop_candidates[f'missing_source_lag_{k}']

            # minimal view for report
            report_cols = ['datetime_beginning_utc', 'hour', 'day_of_week', 'any_warmup', 'any_missing_source']
            drop_report = drop_candidates[report_cols].copy()

            warmup_rows = int(drop_candidates['any_warmup'].sum())
            missing_src_rows = int(drop_candidates['any_missing_source'].sum())
            total_drop_rows = len(drop_candidates)

            print("=== diagnostics: lag-based row drops ===")
            print(f"Total rows (post DA/RT align): {len(combined_data)}")
            if (da_truncated > 0) or (rt_truncated > 0):
                print(f"Truncated due to DA/RT length align -> DA: {da_truncated}, RT: {rt_truncated}")
            print(f"Rows to drop due to missing lag(s): {total_drop_rows}")
            print(f"  of which warm-up rows (first k*24 hours): {warmup_rows}")
            print(f"  of which missing DA source at required lag: {missing_src_rows}")

            # Save detailed report
            try:
                drop_candidates.to_csv('dropped_lag_rows_detailed.csv', index=False)
                drop_report.to_csv('dropped_lag_rows.csv', index=False)
                print("Saved diagnostics: dropped_lag_rows.csv (summary), dropped_lag_rows_detailed.csv (full)")
            except Exception as _:
                pass
        else:
            print("=== diagnostics: lag-based row drops ===")
            print("No rows will be dropped by lag NaNs (rare).")
    except Exception as diag_e:
        print(f"⚠️ diagnostics for lag-based drops failed: {diag_e}")
    
    # 在训练集中显式跳过最早的7天，避免用缺失源数据被填充的滞后特征
    try:
        if not combined_data.empty:
            min_ts = combined_data['datetime_beginning_utc'].min()
            cutoff_ts = min_ts + pd.Timedelta(days=7)
            before_rows = len(combined_data)
            combined_data = combined_data[combined_data['datetime_beginning_utc'] >= cutoff_ts]
            after_rows = len(combined_data)
            print(f"⏭️ skip first 7 days: removed {before_rows - after_rows} rows (cutoff >= {cutoff_ts})")
    except Exception as _:
        # non-fatal; proceed with downstream dropna safeguard
        pass

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


    # 不再对天气聚合特征进行均值填充，后续在特征阶段严格丢弃含 NaN 的样本
    agg_weather_cols = [col for col in combined_data.columns if col.startswith('agg_')]
    
    # add holiday features
    combined_data = apply_holiday_features(combined_data, holidays)
    
    # delete rows with NaN values (due to lag features)
    # The first 7 days will have NaNs due to the 7-day lag features.
    # We should only drop rows that have NaNs in the lag columns.
    lag_cols = [f'total_lmp_da_lag_{lag}' for lag in range(1, 8)]
    # remove temporary diagnostic column if present
    if '__row_idx__' in combined_data.columns:
        # Do not let it affect drop; drop after diagnostics
        pass
    combined_data = combined_data.dropna(subset=lag_cols).reset_index(drop=True)
    if '__row_idx__' in combined_data.columns:
        combined_data.drop(columns=['__row_idx__'], inplace=True)
    
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
        
        # calculate statistical baseline based on time period (use ln-smoothed)
        baseline_stats = data.groupby(['hour', 'day_of_week'])['relative_diff_log'].agg(
            mu='mean',
            sigma='std',
            count='count'
        ).reset_index()
        
        global_sigma = data['relative_diff_log'].std()

        # merge baseline statistics
        merged_data = data.merge(baseline_stats, on=['hour', 'day_of_week'], how='left')
        
        # only keep samples with non-empty mu/sigma and sigma>0
        valid_mask = (~merged_data['mu'].isna()) & (~merged_data['sigma'].isna()) & (merged_data['sigma'] > 0)
        merged_data = merged_data.loc[valid_mask].copy()
        
        # calculate anomaly on ln-smoothed metric
        merged_data['deviation'] = merged_data['relative_diff_log'] - merged_data['mu']
        merged_data['threshold'] = k * merged_data['sigma']
        merged_data['sum'] = merged_data['mu'] + merged_data['threshold']
        merged_data['difference'] = merged_data['mu'] - merged_data['threshold']
        merged_data[['mu', 'threshold','relative_diff_log','sum','difference']].to_csv(f'mu_threshold_k{k}.csv', index=False)
        merged_data['target'] = 0
        merged_data.loc[merged_data['deviation'] > merged_data['threshold'], 'target'] = 1
        merged_data.loc[merged_data['deviation'] < -merged_data['threshold'], 'target'] = 2
        
        # 返回裁剪后的数据（包含原特征与 target）
        data = merged_data
        
    else:  # percentile method
        print(f"=== use percentile method to define anomaly ({percentile}th percentile) ===")
        
        threshold_high = np.percentile(data['relative_diff_log'], percentile)
        threshold_low = np.percentile(data['relative_diff_log'], 100 - percentile)
        data['target'] = 0
        data.loc[data['relative_diff_log'] >= threshold_high, 'target'] = 1
        data.loc[data['relative_diff_log'] <= threshold_low, 'target'] = 2
    
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
    # binary target: anomaly as positive class (1), normal as 0
    y_train = (y_train != 0).astype(int)
    
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
    # Use average_precision to bias training toward better precision/PR under imbalance
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                                  scoring='average_precision', cv=3, n_jobs=-1, verbose=2)

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
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=0.2,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight={0: 5.0, 1: 1.0},
            multi_class='auto',
            max_iter=2000,
            random_state=42
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            class_weight={0: 5.0, 1: 1.0},
            random_state=42
        )
    }
    
    # Start the dictionary of trained models with the optimized RandomForest
    trained_models = {'RandomForest': best_rf_model}
    
    for name, model in models.items():
        print(f"\ntrain {name}...")
        # Train on the training data
        model.fit(X_train_split, y_train_split)
        
        # calculate training set performance (robust to softprob outputs)
        y_pred_train = safe_predict_labels(model, X_train_split)
        train_score = accuracy_score(y_train_split, y_pred_train)
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
    # strictly align feature columns; drop samples with missing features
    available_cols = [col for col in feature_columns if col in test_data.columns]
    X_test = test_data[available_cols].copy()
    # drop samples with missing features
    before_rows = len(X_test)
    valid_mask = ~X_test.isnull().any(axis=1)
    X_test = X_test.loc[valid_mask]
    y_test = test_data.loc[valid_mask, 'target']
    y_test_bin = (y_test != 0).astype(int)
    removed = before_rows - len(X_test)
    if removed > 0:
        print(f"⏭️ drop {removed} test rows due to missing features (no imputation)")
    
    # standardize features
    X_test_scaled = scaler.transform(X_test)
    
    print(f"test set feature shape: {X_test_scaled.shape}")
    print(f"test set target distribution: {y_test.value_counts().to_dict()}")
    
    # test each model
    results = {}
    
    for name, model in trained_models.items():
        print(f"\ntest {name}...")
        
        # 使用异常分数 + 阈值得到最终预测，确保至少有一个正例预测
        y_scores = get_anomaly_scores(model, X_test_scaled)
        anom = compute_anomaly_metrics(model, X_test_scaled, y_test_bin, min_recall=0.01, min_pos_preds=1)
        y_pred = (y_scores >= anom['threshold']).astype(int)

        # calculate evaluation metrics (binary anomaly vs normal)
        accuracy = accuracy_score(y_test_bin, y_pred)
        precision = precision_score(y_test_bin, y_pred, zero_division=0)
        recall = recall_score(y_test_bin, y_pred, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred, zero_division=0)
        
        print(f"{name} test results:")
        print(f"  accuracy: {accuracy:.3f}")
        print(f"  precision: {precision:.3f}")
        print(f"  recall: {recall:.3f}")
        print(f"  F1 score: {f1:.3f}")
        print(f"  anomaly precision: {anom['precision']:.3f} (thr={anom['threshold']:.3f}, pos_preds={anom['pos_preds']})")
        print(f"  anomaly recall:    {anom['recall']:.3f}")
        print(f"  anomaly F1:        {anom['f1']:.3f}")
        
        # confusion matrix
        cm = confusion_matrix(y_test_bin, y_pred)
        print(f"  confusion matrix:\n{cm}")
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'anomaly_precision': anom['precision'],
            'anomaly_recall': anom['recall'],
            'anomaly_f1': anom['f1'],
            'anomaly_threshold': anom['threshold'],
            'anomaly_pos_preds': anom['pos_preds']
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


def find_threshold_max_precision(y_true_binary, y_scores, min_recall=0.01, min_pos_preds=1):
    """
    在给定最小召回与最小正例数量约束下，选择能最大化精确率的阈值；
    若所有阈值召回均为0，则返回高阈值；随后会用 min_pos_preds 强制至少有若干预测。
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, y_scores)
    # precision/recall 长度为 len(thresholds)+1，对齐 thresholds 需去掉最后一个点
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    if len(thresholds) == 0:
        return 1.1  # 强制无正例预测
    mask = recalls >= min_recall
    if not np.any(mask):
        thr_rec = 1.1
    else:
        idx = np.argmax(precisions[mask])
        thr_rec = thresholds[mask][idx]
    # 至少产生 min_pos_preds 个正例：取第 k 大分数为阈值（稍减 epsilon）
    if min_pos_preds is None or min_pos_preds <= 0:
        return float(thr_rec)
    k = int(min_pos_preds)
    k = min(max(k, 1), len(y_scores))
    scores_sorted = np.sort(y_scores)
    thr_pos = scores_sorted[-k]
    thr_pos = np.nextafter(thr_pos, -np.inf)  # 比第k大分数略小
    # 取更严格的阈值以尽量维持高精度，但确保至少有 k 个预测
    thr_final = min(thr_rec, thr_pos)
    return float(thr_final)


def get_anomaly_scores(model, X_mat):
    """
    返回异常分数（二分类为 P(1)，多分类为 P(1)+P(2)）。
    """
    y_scores = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_mat)
        if isinstance(proba, list):
            proba = np.vstack([p for p in proba])
        if proba.ndim == 2 and proba.shape[1] >= 3:
            y_scores = proba[:, 1] + proba[:, 2]
        elif proba.ndim == 2 and proba.shape[1] == 2:
            y_scores = proba[:, 1]
    if y_scores is None and hasattr(model, 'decision_function'):
        df = model.decision_function(X_mat)
        # 标准化到 [0,1]
        y_scores = 1 / (1 + np.exp(-df))
    if y_scores is None:
        # 退化为硬预测
        hard_pred = safe_predict_labels(model, X_mat)
        y_scores = (np.array(hard_pred) != 0).astype(float)
    return y_scores


def compute_anomaly_metrics(model, X_mat, y_true_multiclass, min_recall=0.01, min_pos_preds=1):
    """
    计算“异常”二分类视角的指标（正类=异常：label!=0），阈值通过PR曲线在最小召回约束下最大化精确率。
    返回: dict(precision, recall, f1, pos_preds, threshold)
    """
    y_true_bin = (y_true_multiclass != 0).astype(int)
    y_scores = get_anomaly_scores(model, X_mat)
    thr = find_threshold_max_precision(y_true_bin, y_scores, min_recall=min_recall, min_pos_preds=min_pos_preds)
    y_pred_bin = (y_scores >= thr).astype(int)

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    pos_preds = int(y_pred_bin.sum())
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': float(thr),
        'pos_preds': pos_preds
    }


def safe_predict_labels(model, X_mat):
    """
    兼容不同模型 predict 行为：若返回二维数组（概率或指示矩阵），取 argmax 作为类别标签。
    """
    y_raw = model.predict(X_mat)
    if isinstance(y_raw, np.ndarray) and y_raw.ndim == 2:
        return np.argmax(y_raw, axis=1)
    return y_raw


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


def plot_feature_importance_for_saved_model(top_n=30):
    """
    加载已保存的模型与特征列，绘制并保存 XGBoost 的特征重要性图。
    输出文件：feature_importance_xgb_k{K}.png
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 加载保存的模型与元数据（其中包含 k 值与特征列）
    xgb_model, _scaler, feature_columns, k_value = load_model_for_prediction()

    # 读取特征重要性
    if hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_
    elif hasattr(xgb_model, 'coef_'):
        importances = np.abs(xgb_model.coef_[0])
    else:
        print("当前模型不支持特征重要性提取。")
        return

    feature_names = np.array(feature_columns)
    indices = np.argsort(importances)[-top_n:]

    # 绘图并保存
    plt.figure(figsize=(12, 10))
    plt.title(f'Top {top_n} Feature Importances for XGBoost (k={k_value})')
    plt.barh(range(len(indices)), importances[indices], color='maroon', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    out_path = f'feature_importance_xgb_k{k_value}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")

    # 控制台打印前 top_n 的条目
    print(f"\nTop {top_n} most important features (XGBoost, k={k_value}):")
    for i in np.argsort(importances)[::-1][:top_n]:
        print(f"- {feature_names[i]}: {importances[i]:.4f}")

def test_different_k_values():
    """test different k values"""
    print("=== test different k values ===")
    
    # k values to test
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    # 1. train model (using default k=2.0)
    print("\n=== train model ===")
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
        print(f"\n{'='*20} test k={k} {'='*20}")
        
        # create anomaly target variable using current k value
        test_data_k = create_anomaly_target(test_data.copy(), method='statistical', k=k)
        
        # prepare features (ensure feature order is consistent) - strict, no imputation
        available_cols = [col for col in feature_columns if col in test_data_k.columns]
        X_test = test_data_k[available_cols].copy()
        before_rows = len(X_test)
        valid_mask = ~X_test.isnull().any(axis=1)
        X_test = X_test.loc[valid_mask]
        y_test = test_data_k.loc[valid_mask, 'target']
        
        # standardize features
        X_test_scaled = scaler.transform(X_test)
        
        print(f"test set feature shape: {X_test_scaled.shape}")
        print(f"test set target distribution: {y_test.value_counts().to_dict()}")
        
        # test each model
        k_model_results = {}
        
        for name, model in trained_models.items():
            print(f"\ntest {name}...")
            
            # 使用异常分数 + 阈值得到最终预测，确保至少有一个正例预测
            y_scores = get_anomaly_scores(model, X_test_scaled)
            y_test_bin = (y_test != 0).astype(int)
            anom = compute_anomaly_metrics(model, X_test_scaled, y_test_bin, min_recall=0.01, min_pos_preds=1)
            y_pred = (y_scores >= anom['threshold']).astype(int)

            # calculate evaluation metrics (binary)
            accuracy = accuracy_score(y_test_bin, y_pred)
            precision = precision_score(y_test_bin, y_pred, zero_division=0)
            recall = recall_score(y_test_bin, y_pred, zero_division=0)
            f1 = f1_score(y_test_bin, y_pred, zero_division=0)
            
            print(f"{name} test results:")
            print(f"  accuracy: {accuracy:.3f}")
            print(f"  precision: {precision:.3f}")
            print(f"  recall: {recall:.3f}")
            print(f"  F1 score: {f1:.3f}")
            print(f"  anomaly precision: {anom['precision']:.3f} (thr={anom['threshold']:.3f}, pos_preds={anom['pos_preds']})")
            print(f"  anomaly recall:    {anom['recall']:.3f}")
            print(f"  anomaly F1:        {anom['f1']:.3f}")
            
            # confusion matrix
            cm = confusion_matrix(y_test_bin, y_pred)
            print(f"  confusion matrix:\n{cm}")
            
            k_model_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'anomaly_precision': anom['precision'],
                'anomaly_recall': anom['recall'],
                'anomaly_f1': anom['f1'],
                'anomaly_threshold': anom['threshold'],
                'anomaly_pos_preds': anom['pos_preds'],
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

# =============================
# Saved model evaluation (k=3.5)
# =============================
def evaluate_saved_xgb_on_2025():
    """
    Load the saved XGBoost model and evaluate on 2025 data using the saved k (e.g., k=3.5).
    Outputs:
      - eval_confusion_matrix_k{K}.png
      - eval_anomaly_by_hour_k{K}.png
      - eval_anomaly_by_dow_k{K}.png
      - eval_relative_diff_hist_k{K}.png
      - eval_top_anomalies_k{K}.csv (top 20 highs and lows)
      - prints metrics and relative_diff stats
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    print("\n=== Evaluate saved XGBoost on 2025 ===")
    # 1) load saved artifacts
    xgb_model, scaler, feature_columns, k_value = load_model_for_prediction()
    print(f"Use saved k = {k_value}")

    # 2) load and process 2025 data
    test_data = load_and_process_lmp_data(
        da_file="applicationData/da_hrl_lmps_2025.csv",
        rt_file="applicationData/rt_hrl_lmps_2025.csv",
        weather_dir="weatherData/meteo/"
    )

    # 3) create labels with the saved k
    test_data = create_anomaly_target(test_data, method='statistical', k=k_value)

    # 4) build X in exact training feature order (strict, no imputation)
    available_cols = [col for col in feature_columns if col in test_data.columns]
    X_test = test_data[available_cols].copy()
    before_rows = len(X_test)
    valid_mask = ~X_test.isnull().any(axis=1)
    X_test = X_test.loc[valid_mask]
    y_test = test_data.loc[valid_mask, 'target']

    # 5) scale and predict
    X_test_scaled = scaler.transform(X_test)
    # 使用异常分数 + 阈值得到最终预测，确保至少有一个正例预测
    y_scores = get_anomaly_scores(xgb_model, X_test_scaled)
    y_test_bin = (y_test != 0).astype(int)
    thr = find_threshold_max_precision(y_test_bin, y_scores, min_recall=0.01, min_pos_preds=1)
    y_pred = (y_scores >= thr).astype(int)

    # 6) metrics (binary: anomaly vs normal)
    acc = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, zero_division=0)
    rec = recall_score(y_test_bin, y_pred, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    print("\n=== Overall metrics ===")
    print(f"accuracy: {acc:.3f}")
    print(f"precision (macro): {prec:.3f}")
    print(f"recall (macro): {rec:.3f}")
    print(f"F1 (macro): {f1:.3f}")
    print("\n=== Classification report ===")
    print(classification_report(y_test_bin, y_pred, digits=3, target_names=['Normal','Anomaly']))

    # 7) confusion matrix chart
    cm = confusion_matrix(y_test_bin, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal','Anomaly'],
                yticklabels=['Normal','Anomaly'])
    plt.title(f'Confusion Matrix (XGBoost-binary, k={k_value})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    out_cm = f'eval_confusion_matrix_k{k_value}.png'
    plt.tight_layout()
    plt.savefig(out_cm, dpi=200)
    plt.close()
    print(f"Saved -> {out_cm}")

    # 8) anomaly distributions by hour and DOW
    anomalies = test_data[test_data['target'] != 0].copy()
    # by hour
    plt.figure(figsize=(8,4))
    anomalies['hour'].value_counts().sort_index().plot(kind='bar', color='#c44e52')
    plt.title(f'Anomaly Count by Hour (k={k_value})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    out_hour = f'eval_anomaly_by_hour_k{k_value}.png'
    plt.tight_layout()
    plt.savefig(out_hour, dpi=200)
    plt.close()
    print(f"Saved -> {out_hour}")

    # by day_of_week
    plt.figure(figsize=(8,4))
    dow_counts = anomalies['day_of_week'].value_counts().sort_index()
    dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    # align labels to index
    labels = [dow_labels[i] for i in dow_counts.index]
    plt.bar(labels, dow_counts.values, color='#4c72b0')
    plt.title(f'Anomaly Count by Day of Week (k={k_value})')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    out_dow = f'eval_anomaly_by_dow_k{k_value}.png'
    plt.tight_layout()
    plt.savefig(out_dow, dpi=200)
    plt.close()
    print(f"Saved -> {out_dow}")

    # 9) relative_diff stats and histogram
    rd = test_data['relative_diff']
    print("\n=== relative_diff statistics ===")
    print(f"mean: {rd.mean():.4f}, std: {rd.std():.4f}, max: {rd.max():.4f}, min: {rd.min():.4f}")
    plt.figure(figsize=(8,4))
    plt.hist(rd, bins=60, color='#55a868', alpha=0.9)
    plt.title(f'relative_diff Histogram (k={k_value})')
    plt.xlabel('relative_diff')
    plt.ylabel('Frequency')
    out_hist = f'eval_relative_diff_hist_k{k_value}.png'
    plt.tight_layout()
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"Saved -> {out_hist}")

    # 10) top extremes by relative_diff (high/low)
    top_high = test_data.nlargest(20, 'relative_diff')[['datetime_beginning_utc','total_lmp_da','total_lmp_rt','relative_diff']]
    top_low  = test_data.nsmallest(20, 'relative_diff')[['datetime_beginning_utc','total_lmp_da','total_lmp_rt','relative_diff']]
    top_high['type'] = 'high'
    top_low['type'] = 'low'
    top_all = pd.concat([top_high, top_low], ignore_index=True)
    out_csv = f'eval_top_anomalies_k{k_value}.csv'
    top_all.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    print("\nDone. Charts and CSV are saved in the project root directory.")