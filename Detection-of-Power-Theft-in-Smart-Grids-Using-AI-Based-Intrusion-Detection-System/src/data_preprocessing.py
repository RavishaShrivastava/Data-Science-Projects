"""
Data Preprocessing Module for Power Theft Detection
Handles missing data, feature engineering, and data transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing for smart meter consumption data
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath):
        """Load smart meter data from CSV or Excel"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"Loaded data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df, method='interpolate'):
        """
        Handle missing values using various methods
        
        Args:
            df: DataFrame with missing values
            method: 'interpolate', 'forward_fill', 'backward_fill', 'mean', 'median'
        """
        print(f"Missing values before: {df.isnull().sum().sum()}")
        
        if method == 'interpolate':
            # Time-based interpolation for consumption data
            df = df.interpolate(method='time', limit_direction='both')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        print(f"Missing values after: {df.isnull().sum().sum()}")
        return df
    
    def extract_time_features(self, df, datetime_col='timestamp'):
        """
        Extract time-based features from datetime column
        """
        if datetime_col not in df.columns:
            print(f"Warning: {datetime_col} not found. Skipping time features.")
            return df
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Extract time features
        df['hour'] = df[datetime_col].dt.hour
        df['day'] = df[datetime_col].dt.day
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month
        df['quarter'] = df[datetime_col].dt.quarter
        df['year'] = df[datetime_col].dt.year
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = df['hour'].apply(
            lambda x: 1 if (x >= 18 and x <= 22) or (x >= 6 and x <= 9) else 0
        )
        df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
        
        # Cyclical encoding for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        print(f"Extracted time features. New shape: {df.shape}")
        return df
    
    def extract_consumption_features(self, df, consumption_col='consumption'):
        """
        Extract statistical features from consumption data
        """
        if consumption_col not in df.columns:
            print(f"Warning: {consumption_col} not found. Skipping consumption features.")
            return df
        
        # Rolling window statistics
        for window in [24, 168, 720]:  # 1 day, 1 week, 1 month
            df[f'consumption_mean_{window}h'] = df[consumption_col].rolling(window=window, min_periods=1).mean()
            df[f'consumption_std_{window}h'] = df[consumption_col].rolling(window=window, min_periods=1).std()
            df[f'consumption_max_{window}h'] = df[consumption_col].rolling(window=window, min_periods=1).max()
            df[f'consumption_min_{window}h'] = df[consumption_col].rolling(window=window, min_periods=1).min()
        
        # Rate of change
        df['consumption_diff'] = df[consumption_col].diff()
        df['consumption_pct_change'] = df[consumption_col].pct_change()
        
        # Cumulative features
        df['consumption_cumsum'] = df[consumption_col].cumsum()
        
        # Statistical features
        df['consumption_skew'] = df[consumption_col].rolling(window=168, min_periods=1).skew()
        df['consumption_kurtosis'] = df[consumption_col].rolling(window=168, min_periods=1).kurt()
        
        # Fill NaN values created by rolling operations
        df = df.fillna(method='bfill').fillna(0)
        
        print(f"Extracted consumption features. New shape: {df.shape}")
        return df
    
    def detect_anomalies(self, df, consumption_col='consumption', threshold=3):
        """
        Detect anomalies using statistical methods (Z-score)
        """
        z_scores = np.abs(stats.zscore(df[consumption_col]))
        df['is_anomaly'] = (z_scores > threshold).astype(int)
        df['z_score'] = z_scores
        
        anomaly_count = df['is_anomaly'].sum()
        print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
        
        return df
    
    def create_user_profile_features(self, df, user_id_col='user_id', consumption_col='consumption'):
        """
        Create user-specific profile features
        """
        if user_id_col not in df.columns:
            print(f"Warning: {user_id_col} not found. Skipping user profile features.")
            return df
        
        # Group by user and calculate statistics
        user_stats = df.groupby(user_id_col)[consumption_col].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).reset_index()
        
        user_stats.columns = [user_id_col, 'user_avg_consumption', 'user_std_consumption',
                             'user_min_consumption', 'user_max_consumption', 'user_median_consumption']
        
        # Merge back to original dataframe
        df = df.merge(user_stats, on=user_id_col, how='left')
        
        # Deviation from user's normal behavior
        df['deviation_from_avg'] = df[consumption_col] - df['user_avg_consumption']
        df['deviation_ratio'] = df[consumption_col] / (df['user_avg_consumption'] + 1e-6)
        
        print(f"Created user profile features. New shape: {df.shape}")
        return df
    
    def handle_class_imbalance(self, X, y, method='smote', sampling_strategy=0.5):
        """
        Handle class imbalance using SMOTE or other techniques
        """
        print(f"Class distribution before balancing:\n{pd.Series(y).value_counts()}")
        
        if method == 'smote':
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"Class distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")
            return X_resampled, y_resampled
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """
        Scale features using StandardScaler or MinMaxScaler
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler
    
    def prepare_sequences(self, data, sequence_length=24, target_col='is_theft'):
        """
        Prepare sequential data for LSTM/CNN-LSTM models
        
        Args:
            data: DataFrame with features
            sequence_length: Number of time steps to look back
            target_col: Name of target column
        """
        X, y = [], []
        
        # Separate features and target
        if target_col in data.columns:
            features = data.drop(columns=[target_col]).values
            targets = data[target_col].values
        else:
            features = data.values
            targets = np.zeros(len(data))
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Prepared sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def preprocess_pipeline(self, df, target_col='is_theft', test_size=0.2):
        """
        Complete preprocessing pipeline
        """
        print("=" * 50)
        print("Starting Preprocessing Pipeline")
        print("=" * 50)
        
        # 1. Handle missing values
        df = self.handle_missing_values(df, method='interpolate')
        
        # 2. Extract time features
        if 'timestamp' in df.columns:
            df = self.extract_time_features(df, 'timestamp')
        
        # 3. Extract consumption features
        if 'consumption' in df.columns:
            df = self.extract_consumption_features(df, 'consumption')
        
        # 4. Detect anomalies
        if 'consumption' in df.columns:
            df = self.detect_anomalies(df, 'consumption')
        
        # 5. Create user profile features
        if 'user_id' in df.columns:
            df = self.create_user_profile_features(df, 'user_id', 'consumption')
        
        # 6. Prepare features and target
        # Drop non-numeric and identifier columns
        cols_to_drop = ['timestamp', 'user_id', 'date', 'time']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        if target_col in df.columns:
            X = df.drop(columns=cols_to_drop + [target_col])
            y = df[target_col]
        else:
            X = df.drop(columns=cols_to_drop)
            y = None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # 7. Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # 8. Handle class imbalance
            X_train, y_train = self.handle_class_imbalance(X_train, y_train, method='smote')
            
            # 9. Scale features
            X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
            
            print("=" * 50)
            print("Preprocessing Complete")
            print(f"Training set: {X_train_scaled.shape}")
            print(f"Test set: {X_test_scaled.shape}")
            print("=" * 50)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        else:
            X_scaled, scaler = self.scale_features(X)
            return X_scaled, scaler


def generate_sample_data(n_samples=10000, theft_ratio=0.1):
    """
    Generate sample smart meter data for testing
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    # Generate user IDs
    n_users = 100
    user_ids = np.random.choice(range(1, n_users + 1), size=n_samples)
    
    # Generate normal consumption patterns
    base_consumption = np.random.normal(50, 15, n_samples)
    
    # Add time-based patterns
    hours = timestamps.hour
    consumption = base_consumption * (1 + 0.3 * np.sin(2 * np.pi * hours / 24))
    
    # Add weekly patterns
    day_of_week = timestamps.dayofweek
    consumption = consumption * (1 + 0.1 * (day_of_week < 5))
    
    # Generate theft labels
    n_theft = int(n_samples * theft_ratio)
    is_theft = np.zeros(n_samples)
    theft_indices = np.random.choice(n_samples, n_theft, replace=False)
    is_theft[theft_indices] = 1
    
    # Reduce consumption for theft cases
    consumption[theft_indices] *= np.random.uniform(0.3, 0.7, n_theft)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'user_id': user_ids,
        'consumption': np.maximum(consumption, 0),  # Ensure non-negative
        'is_theft': is_theft.astype(int)
    })
    
    print(f"Generated {n_samples} samples with {n_theft} theft cases ({theft_ratio*100}%)")
    return df


if __name__ == "__main__":
    # Example usage
    print("Generating sample data...")
    df = generate_sample_data(n_samples=10000, theft_ratio=0.1)
    
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_pipeline(df)
    
    print(f"\nFeature names ({len(preprocessor.feature_names)}):")
    print(preprocessor.feature_names[:10], "...")
