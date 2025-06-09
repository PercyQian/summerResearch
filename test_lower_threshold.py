from realtime_prediction import RealTimeLMPPredictor
import numpy as np

def test_different_thresholds():
    """Test different thresholds for anomaly detection"""
    print("Testing different thresholds for anomaly detection")
    print("="*50)
    
    # Create predictor instance
    predictor = RealTimeLMPPredictor()
    
    # Load and process data
    print("Loading data...")
    loaded_data = predictor.load_application_data()
    
    if not loaded_data:
        print("Failed to load data")
        return
    
    print("Processing data...")
    combined_data = predictor.combine_all_data(loaded_data)
    feature_data, feature_names = predictor.create_prediction_features(combined_data)
    
    # Train model
    if predictor.model is None:
        print("Training model...")
        predictor.train_model_from_existing_data()
    
    # Adjust features
    if len(feature_names) != len(predictor.feature_names):
        if len(feature_names) > len(predictor.feature_names):
            feature_data = feature_data.iloc[:, :len(predictor.feature_names)]
        else:
            missing_features = len(predictor.feature_names) - len(feature_names)
            zeros = np.zeros((len(feature_data), missing_features))
            feature_data = np.column_stack([feature_data.values, zeros])
            import pandas as pd
            feature_data = pd.DataFrame(feature_data)
    
    # Test different thresholds
    thresholds = [0.1, 0.15,0.17,0.18, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    print(f"\nData overview:")
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Time range: {combined_data['datetime_utc'].min()} to {combined_data['datetime_utc'].max()}")
    
    print(f"\nThreshold test results:")
    print("-" * 60)
    print(f"{'Threshold':<8} {'Anomaly count':<10} {'Anomaly rate':<10} {'Max probability':<12} {'Average probability':<10}")
    print("-" * 60)
    
    for threshold in thresholds:
        results = predictor.predict_anomalies(feature_data, threshold=threshold)
        
        print(f"{threshold:<8} {results['anomaly_count']:<10} {results['anomaly_rate']:.2%}      "
              f"{results['max_probability']:.4f}       {results['avg_probability']:.4f}")
    
    # Use lower threshold for detailed analysis
    print(f"\nUsing threshold 0.15 for detailed analysis:")
    print("="*50)
    
    results = predictor.predict_anomalies(feature_data, threshold=0.15)
    
    if results['anomaly_count'] > 0:
        anomaly_indices = np.where(results['predictions'] == 1)[0]
        valid_times = combined_data['datetime_utc'].iloc[-len(results['predictions']):]
        
        print(f"Detected {results['anomaly_count']} anomaly time points:")
        for i, idx in enumerate(anomaly_indices[:15]):  # Show first 15
            if idx < len(valid_times):
                timestamp = valid_times.iloc[idx]
                probability = results['probabilities'][idx]
                print(f"  {i+1:2d}. {timestamp}: Anomaly probability {probability:.4f}")
        
        if len(anomaly_indices) > 15:
            print(f"  ... {len(anomaly_indices) - 15} more anomaly points")
            
        # Count anomaly time distribution
        anomaly_hours = []
        for idx in anomaly_indices:
            if idx < len(valid_times):
                hour = valid_times.iloc[idx].hour
                anomaly_hours.append(hour)
        
        print(f"\nAnomaly time distribution:")
        hour_counts = {}
        for hour in anomaly_hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        for hour in sorted(hour_counts.keys()):
            print(f"  {hour:2d}:00 - {hour_counts[hour]} anomalies")
    else:
        print("No anomalies detected with threshold 0.15")

if __name__ == "__main__":
    test_different_thresholds() 