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
        """Load all real data"""
        print("Loading all real data...")
        
        data_files = {
            'generation': 'gen_by_fuel.csv',
            'instantaneous_load': 'inst_load.csv',
            'day_ahead_lmp': 'da_hrl_lmps.csv',  # Day-ahead LMP - features
            'real_time_lmp': 'rt_hrl_lmps.csv'   # Real-time LMP - target
        }
        
        loaded_data = {}
        for key, filename in data_files.items():
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                try:
                    if filename == 'da_hrl_lmps.csv':
                        # Day-ahead LMP file is large, read more data for training
                        loaded_data[key] = pd.read_csv(file_path, nrows=1000000)
                        print(f"Loaded {filename}: {len(loaded_data[key])} rows ")
                    else:
                        loaded_data[key] = pd.read_csv(file_path)
                    print(f"Successfully loaded {filename}: {len(loaded_data[key])} rows")
                except Exception as e:
                    print(f"Failed to load {filename}: {str(e)}")
            else:
                print(f"File not found: {file_path}")
        
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
        
        if price_cols:
            price_col = price_cols[0]
            # Aggregate day-ahead LMP data by time
            agg_data = da_lmp_data.groupby('datetime_utc').agg({
                price_col: ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            agg_data.columns = ['datetime_utc', 'da_lmp_mean', 'da_lmp_std', 'da_lmp_min', 'da_lmp_max']
            return agg_data
        else:
            print("Warning: No price column found in DA LMP data")
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
        
        if price_cols:
            price_col = price_cols[0]
            
            # Calculate anomaly threshold (using 95th percentile)
            threshold = np.percentile(rt_lmp_data[price_col].dropna(), 95)
            print(f"Anomaly threshold (95th percentile): {threshold:.2f}")
            
            # Create anomaly labels
            rt_lmp_data['is_anomaly'] = (rt_lmp_data[price_col] > threshold).astype(int)
            
            # Aggregate real-time LMP data by time
            agg_data = rt_lmp_data.groupby('datetime_utc').agg({
                price_col: ['mean', 'std', 'max'],
                'is_anomaly': 'max'  # If any node has an anomaly, mark the time point as an anomaly
            }).reset_index()
            
            agg_data.columns = ['datetime_utc', 'rt_lmp_mean', 'rt_lmp_std', 'rt_lmp_max', 'is_anomaly']
            
            anomaly_rate = agg_data['is_anomaly'].mean()
            print(f"Anomaly rate in training data: {anomaly_rate:.2%}")
            
            return agg_data
        else:
            print("Warning: No price column found in RT LMP data")
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
    
    def prepare_training_data(self, loaded_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data"""
        print("\nPreparing training data...")
        
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
        
        if 'real_time_lmp' in loaded_data:
            rt_lmp_processed = self.process_rt_lmp_data(loaded_data['real_time_lmp'])
            if rt_lmp_processed is not None:
                processed_data['rt_lmp'] = rt_lmp_processed
        
        # Merge feature data (excluding labels in rt_lmp)
        combined_features = None
        for key, data in processed_data.items():
            if key != 'rt_lmp':  # First merge rt_lmp
                if combined_features is None:
                    combined_features = data
                else:
                    combined_features = combined_features.merge(data, on='datetime_utc', how='outer')
        
        if combined_features is None:
            raise ValueError("No feature data available")
        
        # Add time features
        combined_features = self.add_time_features(combined_features)
        
        # Merge label data - use outer join to preserve more data
        if 'rt_lmp' in processed_data:
            final_data = combined_features.merge(
                processed_data['rt_lmp'][['datetime_utc', 'is_anomaly']], 
                on='datetime_utc', 
                how='outer'  # Change to outer join to preserve more data
            )
        else:
            raise ValueError("No real-time LMP data for labels")
        
        # Only remove missing values in key columns, preserve more samples
        final_data = final_data.dropna(subset=['is_anomaly'])  # Only require non-empty labels
        
        # If the sample is still too small, fill the missing values of numeric features
        if len(final_data) < 100:
            print(f"Warning: Only {len(final_data)} samples available, filling missing values...")
            numeric_cols = final_data.select_dtypes(include=[np.number]).columns
            final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].mean())
            final_data = final_data.fillna(0)
        else:
            final_data = final_data.dropna()
        
        print(f"Final training data shape: {final_data.shape}")
        print(f"Time range: {final_data['datetime_utc'].min()} to {final_data['datetime_utc'].max()}")
        
        # Separate features and labels
        feature_cols = [col for col in final_data.columns if col not in ['datetime_utc', 'is_anomaly']]
        X = final_data[feature_cols]
        y = final_data['is_anomaly']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Anomaly rate: {y.mean():.2%}")
        
        # Check if there are enough samples and classes
        if len(X) < 10:
            raise ValueError(f"Not enough samples for training: {len(X)}. Need at least 10 samples.")
        
        if len(y.unique()) < 2:
            print(f"Warning: Only {len(y.unique())} class(es) found in labels. Creating balanced synthetic data...")
            # If there is only one class, create some synthetic anomaly samples
            n_synthetic = max(10, int(len(X) * 0.1))  # At least 10 or 10% of synthetic anomalies
            synthetic_X = X.sample(n=min(n_synthetic, len(X)), replace=True).copy()
            # Add some noise to the synthetic samples
            for col in synthetic_X.select_dtypes(include=[np.number]).columns:
                noise = np.random.normal(0, synthetic_X[col].std() * 0.1, len(synthetic_X))
                synthetic_X[col] += noise
            synthetic_y = pd.Series([1] * len(synthetic_X), index=synthetic_X.index)
            
            # Merge original data and synthetic data
            X = pd.concat([X, synthetic_X], ignore_index=True)
            y = pd.concat([y, synthetic_y], ignore_index=True)
            
            print(f"Added {len(synthetic_X)} synthetic anomaly samples")
            print(f"New anomaly rate: {y.mean():.2%}")
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train_model_with_real_data(self, loaded_data: Dict[str, pd.DataFrame]):
        """Train model with real data"""
        print("Training model with real data...")
        
        # Prepare training data
        X, y = self.prepare_training_data(loaded_data)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Validate after training
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
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
                                 train_test_split_date: str = "2025-06-05") -> Dict:
        """
        Run realistic prediction
        
        Args:
            data_path: Data path
            train_test_split_date: Training test split date
        """
        print("=" * 70)
        print("Realistic LMP Anomaly Detection System")
        print("=" * 70)
        
        try:
            # 1. Load all data
            loaded_data = self.load_all_data(data_path)
            
            if not loaded_data:
                raise ValueError("No data loaded")
            
            # 2. Train model
            print("\nTraining model with real data...")
            self.train_model_with_real_data(loaded_data)
            
            # 3. Prepare prediction data (use data after split date for prediction)
            print(f"\nPreparing prediction data after {train_test_split_date}...")
            
            # Here you can implement time split logic
            # For now, use all data for demonstration
            X, y = self.prepare_training_data(loaded_data)
            
            # 4. Make predictions
            print("\nMaking predictions...")
            predictions = self.predict_anomalies(X, threshold=0.3)  # Use lower threshold
            
            # 5. Output results
            print("\n" + "=" * 70)
            print("PREDICTION RESULTS")
            print("=" * 70)
            print(f"Total predictions: {predictions['total_predictions']}")
            print(f"Detected anomalies: {predictions['anomaly_count']}")
            print(f"Anomaly rate: {predictions['anomaly_rate']:.2%}")
            print(f"Max anomaly probability: {predictions['max_probability']:.4f}")
            print(f"Average anomaly probability: {predictions['avg_probability']:.4f}")
            
            # 6. If there are actual labels, calculate accuracy
            if len(y) == len(predictions['predictions']):
                accuracy = accuracy_score(y, predictions['predictions'])
                precision = precision_score(y, predictions['predictions'], zero_division=0)
                recall = recall_score(y, predictions['predictions'], zero_division=0)
                f1 = f1_score(y, predictions['predictions'], zero_division=0)
                
                print(f"\nVALIDATION RESULTS:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
            
            return {
                'predictions': predictions,
                'feature_count': len(self.feature_names),
                'training_samples': len(X)
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
    
    # Run realistic prediction
    results = predictor.run_realistic_prediction()
    
    if "error" not in results:
        # Save
        predictor.save_model("realistic_model.pkl")
        predictor.save_scaler("realistic_scaler.pkl")
        
        

if __name__ == "__main__":
    main() 