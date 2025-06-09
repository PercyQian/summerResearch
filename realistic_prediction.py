import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import datetime as dt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RealisticLMPPredictor:
    """
    Realistic LMP anomaly detection prediction system
    Correctly use da_hrl_lmps and rt_hrl_lmps data
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
    
    def load_all_data(self, data_path: str = "applicationData") -> Dict[str, pd.DataFrame]:
        """Load all real data including multiple DA and RT LMP files"""
        print("Loading all real data...")
        
        # First load the basic files
        data_files = {
            'generation': 'gen_by_fuel.csv',
            'instantaneous_load': 'inst_load.csv'
        }
        
        loaded_data = {}
        for key, filename in data_files.items():
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                try:
                    loaded_data[key] = pd.read_csv(file_path)
                    print(f"Successfully loaded {filename}: {len(loaded_data[key])} rows")
                except Exception as e:
                    print(f"Failed to load {filename}: {str(e)}")
            else:
                print(f"File not found: {file_path}")
        
        # Load multiple DA LMP files 
        da_lmp_files = ['da_hrl_lmps_5.30.csv','da_hrl_lmps_5.31.csv','da_hrl_lmps_6.1.csv', 'da_hrl_lmps_6.2.csv', 'da_hrl_lmps_6.3.csv', 'da_hrl_lmps_6.4.csv', 'da_hrl_lmps_6.5.csv']
        da_lmp_data_list = []
        
        for filename in da_lmp_files:
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    da_lmp_data_list.append(df)
                    print(f"Successfully loaded {filename}: {len(df)} rows")
                except Exception as e:
                    print(f"Failed to load {filename}: {str(e)}")
            else:
                print(f"DA LMP file not found: {file_path}")
        
        # Combine all DA LMP data
        if da_lmp_data_list:
            combined_da_lmp = pd.concat(da_lmp_data_list, ignore_index=True)
            loaded_data['day_ahead_lmp'] = combined_da_lmp
            print(f"Combined DA LMP data: {len(combined_da_lmp)} rows total")
        else:
            print("Warning: No DA LMP data files found")
        
        # Load multiple RT LMP files 
        rt_lmp_files = ['rt_hrl_lmps_5.30.csv','rt_hrl_lmps_5.31.csv','rt_hrl_lmps_6.1.csv','rt_hrl_lmps_6.2.csv', 'rt_hrl_lmps_6.3.csv', 'rt_hrl_lmps_6.4.csv', 'rt_hrl_lmps_6.5.csv']
        rt_lmp_data_list = []
        
        for filename in rt_lmp_files:
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    rt_lmp_data_list.append(df)
                    print(f"Successfully loaded {filename}: {len(df)} rows")
                except Exception as e:
                    print(f"Failed to load {filename}: {str(e)}")
            else:
                print(f"RT LMP file not found: {file_path}")
        
        # Combine all RT LMP data
        if rt_lmp_data_list:
            combined_rt_lmp = pd.concat(rt_lmp_data_list, ignore_index=True)
            loaded_data['real_time_lmp'] = combined_rt_lmp
            print(f"Combined RT LMP data: {len(combined_rt_lmp)} rows total")
        else:
            print("Warning: No RT LMP data files found")
        
        return loaded_data
    
    def process_generation_data(self, gen_data: pd.DataFrame) -> pd.DataFrame:
        """Process generation data"""
        gen_data['datetime_utc'] = pd.to_datetime(gen_data['datetime_beginning_utc'])
        
        # Aggregate generation data by time
        agg_data = gen_data.groupby('datetime_utc').agg({
            'mw': ['sum', 'mean', 'std'],
            'fuel_percentage_of_total': ['mean', 'std']
        }).reset_index()
        
        agg_data.columns = ['datetime_utc', 'total_generation_mw', 'avg_generation_mw', 
                           'std_generation_mw', 'avg_fuel_percentage', 'std_fuel_percentage']
        
        # Calculate renewable energy ratio
        renewable_data = gen_data[gen_data['is_renewable'] == True].groupby('datetime_utc')['mw'].sum().reset_index()
        renewable_data.columns = ['datetime_utc', 'renewable_mw']
        
        result = agg_data.merge(renewable_data, on='datetime_utc', how='left')
        result['renewable_percentage'] = result['renewable_mw'] / result['total_generation_mw']
        result['renewable_percentage'] = result['renewable_percentage'].fillna(0)
        
        return result
    
    def process_load_data(self, load_data: pd.DataFrame) -> pd.DataFrame:
        """Process load data"""
        load_data['datetime_utc'] = pd.to_datetime(load_data['datetime_beginning_utc'])
        
        agg_data = load_data.groupby('datetime_utc').agg({
            'instantaneous_load': ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        agg_data.columns = ['datetime_utc', 'total_inst_load', 'avg_inst_load', 
                           'std_inst_load', 'min_inst_load', 'max_inst_load']
        
        return agg_data
    
    def process_da_lmp_data(self, da_lmp_data: pd.DataFrame) -> pd.DataFrame:
        """Process day-ahead LMP data as features"""
        print("Processing day-ahead LMP data as features...")
        
       
        print(f"DA LMP columns: {da_lmp_data.columns.tolist()}")
        
        # Try to find the correct time and price columns
        time_cols = [col for col in da_lmp_data.columns if 'datetime' in col.lower() or 'time' in col.lower()]
        price_cols = [col for col in da_lmp_data.columns if 'lmp' in col.lower() or 'price' in col.lower()]
        
        print(f"Time columns found: {time_cols}")
        print(f"Price columns found: {price_cols}")
        
        if time_cols:
            time_col = time_cols[0]
            da_lmp_data['datetime_utc'] = pd.to_datetime(da_lmp_data[time_col])
        else:
            print("Warning: No datetime column found in DA LMP data")
            return None
        
        # Explicitly use total_lmp_da column
        if 'total_lmp_da' in da_lmp_data.columns:
            price_col = 'total_lmp_da'
            print(f"Using DA LMP column: {price_col}")
            
            # Aggregate day-ahead LMP data by time
            agg_data = da_lmp_data.groupby('datetime_utc').agg({
                price_col: ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            agg_data.columns = ['datetime_utc', 'da_lmp_mean', 'da_lmp_std', 'da_lmp_min', 'da_lmp_max']
            return agg_data
        else:
            print("Warning: total_lmp_da column not found in DA LMP data")
            print(f"Available price columns: {price_cols}")
            return None
    
    def process_rt_lmp_data(self, rt_lmp_data: pd.DataFrame) -> pd.DataFrame:
        """Process real-time LMP data to create labels"""
        print("Processing real-time LMP data to create labels...")
        
        # Check column names
        print(f"RT LMP columns: {rt_lmp_data.columns.tolist()}")
        
        # Try to find the correct time and price columns
        time_cols = [col for col in rt_lmp_data.columns if 'datetime' in col.lower() or 'time' in col.lower()]
        price_cols = [col for col in rt_lmp_data.columns if 'lmp' in col.lower() or 'price' in col.lower()]
        
        print(f"Time columns found: {time_cols}")
        print(f"Price columns found: {price_cols}")
        
        if time_cols:
            time_col = time_cols[0]
            rt_lmp_data['datetime_utc'] = pd.to_datetime(rt_lmp_data[time_col])
        else:
            print("Warning: No datetime column found in RT LMP data")
            return None
        
        # Explicitly use total_lmp_rt column
        if 'total_lmp_rt' in rt_lmp_data.columns:
            price_col = 'total_lmp_rt'
            print(f"Using RT LMP column: {price_col}")
            
            # DON'T calculate threshold here - will be calculated after time split
            # Just aggregate the data first
            agg_data = rt_lmp_data.groupby('datetime_utc').agg({
                price_col: ['mean', 'std', 'max']
            }).reset_index()
            
            agg_data.columns = ['datetime_utc', 'rt_lmp_mean', 'rt_lmp_std', 'rt_lmp_max']
            
            # Add the raw price column for threshold calculation later
            agg_data['rt_lmp_price'] = agg_data['rt_lmp_mean']  # Use mean as representative price
            
            print(f"RT LMP data aggregated: {len(agg_data)} time points")
            
            return agg_data
        else:
            print("Warning: total_lmp_rt column not found in RT LMP data")
            print(f"Available price columns: {price_cols}")
            return None
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time features"""
        data = data.copy()
        data['hour'] = data['datetime_utc'].dt.hour
        data['day_of_week'] = data['datetime_utc'].dt.dayofweek
        data['month'] = data['datetime_utc'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Add key time period identifiers
        data['is_peak_morning'] = ((data['hour'] >= 6) & (data['hour'] <= 10)).astype(int)
        data['is_peak_evening'] = ((data['hour'] >= 17) & (data['hour'] <= 21)).astype(int)
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 5)).astype(int)
        
        return data
    
    def prepare_training_data_with_da_focus(self, loaded_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data with DA LMP focus:
        - Use historical DA LMP patterns to predict DA LMP anomalies
        - This allows us to use 6/02-6/04 DA data for training
        """
        print("\nPreparing training data with DA LMP focus...")
        
        # Process DA LMP data as both features and target
        if 'day_ahead_lmp' not in loaded_data:
            raise ValueError("No day-ahead LMP data available")
        
        da_lmp_data = loaded_data['day_ahead_lmp'].copy()
        
        # Convert datetime
        da_lmp_data['datetime_utc'] = pd.to_datetime(da_lmp_data['datetime_beginning_utc'])
        
        # Use total_lmp_da column
        if 'total_lmp_da' not in da_lmp_data.columns:
            raise ValueError("total_lmp_da column not found")
        
        print(f"Processing DA LMP data: {len(da_lmp_data)} rows")
        print(f"Time range: {da_lmp_data['datetime_utc'].min()} to {da_lmp_data['datetime_utc'].max()}")
        
        # Aggregate by hour to get hourly features
        hourly_data = da_lmp_data.groupby('datetime_utc').agg({
            'total_lmp_da': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        hourly_data.columns = ['datetime_utc', 'lmp_mean', 'lmp_std', 'lmp_min', 'lmp_max', 'lmp_count']
        
        # Add time-based features
        hourly_data = self.add_time_features(hourly_data)
        
        # Create lag features (use previous hours to predict current hour)
        lag_hours = [1, 2, 3, 6, 12, 24]  # 1h, 2h, 3h, 6h, 12h, 24h ago
        
        for lag in lag_hours:
            hourly_data[f'lmp_mean_lag_{lag}h'] = hourly_data['lmp_mean'].shift(lag)
            hourly_data[f'lmp_max_lag_{lag}h'] = hourly_data['lmp_max'].shift(lag)
        
        # Create rolling statistics
        hourly_data['lmp_mean_roll_24h'] = hourly_data['lmp_mean'].rolling(window=24, min_periods=1).mean()
        hourly_data['lmp_std_roll_24h'] = hourly_data['lmp_mean'].rolling(window=24, min_periods=1).std()
        hourly_data['lmp_max_roll_24h'] = hourly_data['lmp_max'].rolling(window=24, min_periods=1).max()
        
        # Remove rows with too many NaN values (due to lag features)
        hourly_data = hourly_data.dropna()
        
        print(f"After feature engineering: {len(hourly_data)} samples")
        print(f"Time range: {hourly_data['datetime_utc'].min()} to {hourly_data['datetime_utc'].max()}")
        
        # Separate features and target
        feature_cols = [col for col in hourly_data.columns 
                       if col not in ['datetime_utc', 'lmp_mean']]  # Use lmp_mean as target
        
        X = hourly_data[feature_cols]
        target_prices = hourly_data['lmp_mean']  # Target variable
        datetime_col = hourly_data['datetime_utc']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Target price range: ${target_prices.min():.2f} - ${target_prices.max():.2f}")
        
        self.feature_names = feature_cols
        
        return X, target_prices, datetime_col
    
    def prepare_training_data(self, loaded_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data - enhanced version"""
        
        # Check what data we have
        has_full_rt_data = False
        if 'real_time_lmp' in loaded_data:
            rt_data = loaded_data['real_time_lmp']
            rt_times = pd.to_datetime(rt_data['datetime_beginning_utc'])
            earliest_rt = rt_times.min()
            print(f"RT LMP data starts from: {earliest_rt}")
            
            # Check if RT data covers our intended training period (before 2025-06-05)
            training_cutoff = pd.to_datetime("2025-06-05")
            has_full_rt_data = earliest_rt < training_cutoff
        
        if has_full_rt_data:
            print("Using original strategy: RT LMP as target")
            return self.prepare_training_data_original(loaded_data)
        else:
            print("Using DA LMP strategy: Historical DA patterns to predict DA anomalies")
            return self.prepare_training_data_with_da_focus(loaded_data)
    
    def prepare_training_data_original(self, loaded_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Original training data preparation method"""
        print("\nPreparing training data (original method)...")
        
        # Process various data
        processed_data = {}
        
        if 'generation' in loaded_data:
            processed_data['generation'] = self.process_generation_data(loaded_data['generation'])
        
        if 'instantaneous_load' in loaded_data:
            processed_data['load'] = self.process_load_data(loaded_data['instantaneous_load'])
        
        if 'day_ahead_lmp' in loaded_data:
            da_lmp_processed = self.process_da_lmp_data(loaded_data['day_ahead_lmp'])
            if da_lmp_processed is not None:
                processed_data['da_lmp'] = da_lmp_processed
        
        # Check if we have RT LMP data for the full time range
        rt_lmp_processed = None
        if 'real_time_lmp' in loaded_data:
            rt_lmp_processed = self.process_rt_lmp_data(loaded_data['real_time_lmp'])
            if rt_lmp_processed is not None:
                processed_data['rt_lmp'] = rt_lmp_processed

        # Merge feature data (excluding labels)
        combined_features = None
        for key, data in processed_data.items():
            if key != 'rt_lmp':  # First merge features (not labels)
                if combined_features is None:
                    combined_features = data
                else:
                    combined_features = combined_features.merge(data, on='datetime_utc', how='outer')
        
        if combined_features is None:
            raise ValueError("No feature data available")
        
        # Add time features
        combined_features = self.add_time_features(combined_features)
        
        # Strategy: If we don't have RT LMP for early dates, use DA LMP data itself for training
        # This becomes a "predict DA LMP anomalies based on historical DA LMP patterns" task
        
        if rt_lmp_processed is not None:
            # Original strategy: use RT LMP as labels where available
            final_data = combined_features.merge(
                processed_data['rt_lmp'][['datetime_utc', 'rt_lmp_price']], 
                on='datetime_utc', 
                how='outer'
            )
            final_data = final_data.dropna(subset=['rt_lmp_price'])
            price_column = 'rt_lmp_price'
            print("Using RT LMP as target variable")
        else:
            # Fallback strategy: use DA LMP as both features and target
            final_data = combined_features.copy()
            if 'da_lmp_mean' in final_data.columns:
                final_data['target_price'] = final_data['da_lmp_mean']
                price_column = 'target_price'
                print("Using DA LMP as target variable (RT LMP not available for full range)")
            else:
                raise ValueError("No price data available for creating labels")
        
        print(f"Final training data shape: {final_data.shape}")
        print(f"Time range: {final_data['datetime_utc'].min()} to {final_data['datetime_utc'].max()}")
        
        # Separate features, prices, and datetime  
        feature_cols = [col for col in final_data.columns if col not in ['datetime_utc', price_column]]
        X = final_data[feature_cols]
        prices = final_data[price_column]  # Will be used to calculate anomaly labels after time split
        datetime_col = final_data['datetime_utc']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        # Check if there are enough samples
        if len(X) < 10:
            raise ValueError(f"Not enough samples for training: {len(X)}. Need at least 10 samples.")
        
        self.feature_names = feature_cols
        
        # Return features, prices (for later threshold calculation), and datetime
        return X, prices, datetime_col
    
    def train_model_with_real_data(self, X_train, y_train):
        """Train model with real data - updated to accept preprocessed training data"""
        print("Training model with real data...")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y_train)
        
        # Validate after training
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, zero_division=0)
        recall = recall_score(y_train, y_pred, zero_division=0)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        
        print(f"\nTraining results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 most important features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def predict_anomalies(self, feature_data: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Predict anomalies"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure feature order is correct
        feature_data_ordered = feature_data[self.feature_names]
        
        # Standardize features
        X_scaled = self.scaler.transform(feature_data_ordered)
        
        # Predict
        try:
            prob_result = self.model.predict_proba(X_scaled)
            # Check if there are two classes
            if prob_result.shape[1] == 2:
                probabilities = prob_result[:, 1]  # Probability of anomaly class
            else:
                # If there is only one class
                probabilities = np.zeros(len(X_scaled))  # Default all predictions to normal
                print("Warning: Model only learned one class, defaulting to normal predictions")
        except Exception as e:
            print(f"Prediction error: {e}")
            # If prediction fails, return default value
            probabilities = np.random.uniform(0.1, 0.3, len(X_scaled))  # Random low probability
        
        predictions = (probabilities >= threshold).astype(int)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'anomaly_count': np.sum(predictions),
            'total_predictions': len(predictions),
            'anomaly_rate': np.mean(predictions),
            'max_probability': np.max(probabilities) if len(probabilities) > 0 else 0,
            'avg_probability': np.mean(probabilities) if len(probabilities) > 0 else 0
        }
    
    def run_realistic_prediction(self, data_path: str = "applicationData", 
                                 train_end_date: str = "2025-06-05", 
                                 val_end_date: str = "2025-06-06") -> Dict:
        """
        Run realistic prediction with train/validation/test split
        
        Args:
            data_path: Data path
            train_end_date: End date for training (start of validation)
            val_end_date: End date for validation (start of testing)
        """
        print("=" * 80)
        print("LMP Anomaly Detection: Train/Validation/Test Split")
        print("=" * 80)
        
        try:
            # 1. Load all data
            loaded_data = self.load_all_data(data_path)
            
            if not loaded_data:
                raise ValueError("No data loaded")
            
            # 2. Prepare ALL data with features, prices, and datetime
            print("\nPreparing all data...")
            X_all, prices_all, datetime_all = self.prepare_training_data(loaded_data)
            
            # 3. Implement three-way time-based split
            print(f"\nImplementing three-way time split:")
            print(f"  Training: < {train_end_date}")
            print(f"  Validation: {train_end_date} to {val_end_date}")
            print(f"  Testing: >= {val_end_date}")
            
            train_end = pd.to_datetime(train_end_date)
            val_end = pd.to_datetime(val_end_date)
            
            # Create train/val/test masks
            mask_train = datetime_all < train_end
            mask_val = (datetime_all >= train_end) & (datetime_all < val_end)
            mask_test = datetime_all >= val_end
            
            # Split data
            X_train, X_val, X_test = X_all[mask_train], X_all[mask_val], X_all[mask_test]
            prices_train = prices_all[mask_train]
            prices_val = prices_all[mask_val] 
            prices_test = prices_all[mask_test]
            datetime_train = datetime_all[mask_train]
            datetime_val = datetime_all[mask_val]
            datetime_test = datetime_all[mask_test]
            
            print(f"\nData split summary:")
            print(f"Training: {len(X_train)} samples ({datetime_train.min()} to {datetime_train.max()})")
            print(f"Validation: {len(X_val)} samples ({datetime_val.min()} to {datetime_val.max()})")
            print(f"Testing: {len(X_test)} samples ({datetime_test.min()} to {datetime_test.max()})")
            
            if len(X_train) == 0:
                raise ValueError("No training data available")
            if len(X_val) == 0:
                raise ValueError("No validation data available")
            if len(X_test) == 0:
                raise ValueError("No testing data available")
            
            # 4. Calculate anomaly threshold ONLY on training data
            print(f"\nCalculating anomaly threshold on training data only...")
            threshold = np.percentile(prices_train.dropna(), 95)
            print(f"Initial threshold (95th percentile of training data): ${threshold:.2f}")
            
            # Create labels based on training threshold
            y_train = (prices_train > threshold).astype(int)
            y_val = (prices_val > threshold).astype(int)
            y_test = (prices_test > threshold).astype(int)
            
            print(f"Training anomaly rate: {y_train.mean():.2%}")
            print(f"Validation anomaly rate: {y_val.mean():.2%}")
            print(f"Testing anomaly rate: {y_test.mean():.2%}")
            
            # 5. Train model ONLY on training data
            print(f"\nTraining model on {len(X_train)} samples...")
            if len(y_train.unique()) < 2:
                print(f"Warning: Only {len(y_train.unique())} class(es) in training data. Adjusting threshold...")
                threshold = np.percentile(prices_train.dropna(), 90)
                print(f"Adjusted threshold to 90th percentile: ${threshold:.2f}")
                y_train = (prices_train > threshold).astype(int)
                y_val = (prices_val > threshold).astype(int)
                y_test = (prices_test > threshold).astype(int)
            
            self.feature_names = X_train.columns.tolist()
            self.train_model_with_real_data(X_train, y_train)
            
            # 6. Optimize threshold on VALIDATION data
            print(f"\n" + "="*60)
            print("VALIDATION SET - THRESHOLD OPTIMIZATION")
            print("="*60)
            
            X_val_scaled = self.scaler.transform(X_val)
            val_probabilities = self.model.predict_proba(X_val_scaled)[:, 1]
            
            # Test different thresholds on validation set
            thresholds = [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            best_f1 = -1
            best_threshold = 0.5
            best_val_predictions = None
            
            print(f"\nTesting thresholds on VALIDATION data:")
            print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Pred_Anom':<10}")
            print("-" * 70)
            
            for threshold_test in thresholds:
                val_predictions = (val_probabilities >= threshold_test).astype(int)
                
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_precision = precision_score(y_val, val_predictions, zero_division=0)
                val_recall = recall_score(y_val, val_predictions, zero_division=0)
                val_f1 = f1_score(y_val, val_predictions, zero_division=0)
                
                print(f"{threshold_test:<10.1f} {val_accuracy:<10.3f} {val_precision:<10.3f} {val_recall:<10.3f} {val_f1:<10.3f} {val_predictions.sum():<10}")
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_threshold = threshold_test
                    best_val_predictions = val_predictions
            
            print(f"\n✅ Best threshold from validation: {best_threshold} (F1: {best_f1:.3f})")
            
            # 7. Final evaluation on TEST data (never seen before)
            print(f"\n" + "="*60)
            print("FINAL TEST SET EVALUATION")
            print("="*60)
            
            X_test_scaled = self.scaler.transform(X_test)
            test_probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Use the best threshold from validation
            final_test_predictions = (test_probabilities >= best_threshold).astype(int)
            
            # Calculate final metrics
            final_accuracy = accuracy_score(y_test, final_test_predictions)
            final_precision = precision_score(y_test, final_test_predictions, zero_division=0)
            final_recall = recall_score(y_test, final_test_predictions, zero_division=0)
            final_f1 = f1_score(y_test, final_test_predictions, zero_division=0)
            
            print(f"\nFINAL TEST RESULTS (Never seen before):")
            print(f"✅ Test samples: {len(X_test)}")
            print(f"✅ Optimal threshold: {best_threshold}")
            print(f"✅ Predicted anomalies: {final_test_predictions.sum()}")
            print(f"✅ Actual anomalies: {y_test.sum()}")
            print(f"✅ Accuracy: {final_accuracy:.4f}")
            print(f"✅ Precision: {final_precision:.4f}")
            print(f"✅ Recall: {final_recall:.4f}")
            print(f"✅ F1 Score: {final_f1:.4f}")
            
            # Show some prediction examples
            print(f"\nPrediction examples from test set:")
            for i in range(min(10, len(X_test))):
                actual = "Anomaly" if y_test.iloc[i] == 1 else "Normal"
                predicted = "Anomaly" if final_test_predictions[i] == 1 else "Normal"
                prob = test_probabilities[i]
                price = prices_test.iloc[i]
                time = datetime_test.iloc[i]
                print(f"  {time}: ${price:.2f} | Prob: {prob:.3f} | Actual: {actual:<7} | Pred: {predicted}")
            
            # Feature importance analysis
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 10 most important features:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Performance comparison across sets
            print(f"\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            
            # Training performance
            train_pred = self.model.predict(self.scaler.transform(X_train))
            train_f1 = f1_score(y_train, train_pred, zero_division=0)
            
            print(f"Training F1:   {train_f1:.4f} (reference only)")
            print(f"Validation F1: {best_f1:.4f} (used for optimization)")
            print(f"Test F1:       {final_f1:.4f} (final performance)")
            
            return {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'optimal_threshold': best_threshold,
                'train_f1': train_f1,
                'val_f1': best_f1,
                'test_f1': final_f1,
                'test_accuracy': final_accuracy,
                'test_precision': final_precision,
                'test_recall': final_recall,
                'predicted_anomalies': final_test_predictions.sum(),
                'actual_anomalies': y_test.sum(),
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def save_model(self, model_path: str):
        """Save model"""
        if self.model is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {model_path}")
    
    def save_scaler(self, scaler_path: str):
        """Save scaler"""
        if self.scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {scaler_path}")

def main():
    """Main function"""
    predictor = RealisticLMPPredictor()
    
    # Run realistic prediction with new strategy: Train(6/2-6/3) / Val(6/4) / Test(6/5)
    results = predictor.run_realistic_prediction(train_end_date="2025-06-04", val_end_date="2025-06-05")
    
    if "error" not in results:
        # Save
        predictor.save_model("realistic_model.pkl")
        predictor.save_scaler("realistic_scaler.pkl")
        

if __name__ == "__main__":
    main() 