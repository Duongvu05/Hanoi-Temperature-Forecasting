"""
Weather Data Processing Module

This module provides comprehensive data processing capabilities for weather data,
including cleaning, validation, transformation, and feature preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import yaml
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDataProcessor:
    """
    A comprehensive class for processing weather data.
    
    This class handles:
    - Data validation and quality checks
    - Missing value treatment
    - Outlier detection and handling
    - Feature type conversion
    - Data normalization and scaling
    - Feature filtering and selection
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialize the Weather Data Processor.
        
        Args:
            config_path: Path to data configuration file.
        """
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            self.config = self._get_default_config()
        
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not found."""
        return {
            'processing': {
                'validation': {
                    'check_date_continuity': True,
                    'check_missing_values': True,
                    'check_outliers': True,
                    'outlier_method': 'iqr',
                    'outlier_threshold': 3.0
                },
                'missing_values': {
                    'strategy': 'interpolate',
                    'interpolation_method': 'linear',
                    'max_consecutive_missing': 3
                },
                'type_conversion': {
                    'datetime_columns': ['datetime', 'sunrise', 'sunset'],
                    'numeric_columns': ['temp', 'tempmax', 'tempmin', 'humidity', 'pressure'],
                    'categorical_columns': ['conditions', 'description', 'icon']
                }
            },
            'quality_checks': {
                'temperature': {'min_value': -20.0, 'max_value': 50.0},
                'humidity': {'min_value': 0.0, 'max_value': 100.0},
                'pressure': {'min_value': 950.0, 'max_value': 1050.0}
            }
        }
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate the input data and return validation report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (validated_df, validation_report)
        """
        logger.info("Starting data validation...")
        validation_report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'validation_checks': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # Check basic data structure
        if df.empty:
            validation_report['issues_found'].append("Dataset is empty")
            return df, validation_report
        
        # Check for required datetime column
        if 'datetime' not in df.columns:
            validation_report['issues_found'].append("Missing required 'datetime' column")
        else:
            # Validate datetime continuity
            if self.config['processing']['validation']['check_date_continuity']:
                date_gaps = self._check_date_continuity(df)
                if date_gaps:
                    validation_report['validation_checks']['date_gaps'] = len(date_gaps)
                    validation_report['issues_found'].append(f"Found {len(date_gaps)} date gaps")
        
        # Check missing values
        if self.config['processing']['validation']['check_missing_values']:
            missing_report = self._check_missing_values(df)
            validation_report['validation_checks']['missing_values'] = missing_report
            
            total_missing = sum(missing_report.values())
            if total_missing > 0:
                validation_report['issues_found'].append(f"Found {total_missing} missing values")
        
        # Check for outliers
        if self.config['processing']['validation']['check_outliers']:
            outliers_report = self._check_outliers(df)
            validation_report['validation_checks']['outliers'] = outliers_report
            
            total_outliers = sum(outliers_report.values())
            if total_outliers > 0:
                validation_report['issues_found'].append(f"Found {total_outliers} potential outliers")
        
        # Quality checks for specific features
        quality_issues = self._perform_quality_checks(df)
        if quality_issues:
            validation_report['validation_checks']['quality_issues'] = quality_issues
            validation_report['issues_found'].extend(quality_issues)
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(validation_report)
        
        logger.info(f"Data validation completed. Found {len(validation_report['issues_found'])} issues.")
        return df, validation_report
    
    def _check_date_continuity(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Check for gaps in date continuity."""
        if 'datetime' not in df.columns:
            return []
        
        df_sorted = df.sort_values('datetime')
        df_sorted['datetime'] = pd.to_datetime(df_sorted['datetime'])
        
        # Calculate expected frequency (daily or hourly)
        time_diffs = df_sorted['datetime'].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(days=1)
        
        # Find gaps larger than expected
        gaps = []
        for i, diff in enumerate(time_diffs):
            if diff > most_common_diff * 1.5:  # Allow some tolerance
                start_date = df_sorted.iloc[i]['datetime']
                end_date = df_sorted.iloc[i + 1]['datetime']
                gaps.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        return gaps
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values in the dataset."""
        missing_counts = df.isnull().sum()
        return {col: count for col, count in missing_counts.items() if count > 0}
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for outliers in numerical columns."""
        outliers_count = {}
        method = self.config['processing']['validation']['outlier_method']
        threshold = self.config['processing']['validation']['outlier_threshold']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in df.columns and not df[col].empty:
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(df[col])
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(df[col], threshold)
                else:
                    outliers = []
                
                outliers_count[col] = len(outliers)
        
        return outliers_count
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold].index.tolist()
        return outliers
    
    def _perform_quality_checks(self, df: pd.DataFrame) -> List[str]:
        """Perform quality checks on specific weather features."""
        issues = []
        quality_config = self.config.get('quality_checks', {})
        
        # Temperature checks
        if 'temp' in df.columns and 'temperature' in quality_config:
            temp_config = quality_config['temperature']
            invalid_temps = df[
                (df['temp'] < temp_config['min_value']) | 
                (df['temp'] > temp_config['max_value'])
            ]
            if len(invalid_temps) > 0:
                issues.append(f"Found {len(invalid_temps)} temperature values outside valid range")
        
        # Humidity checks
        if 'humidity' in df.columns and 'humidity' in quality_config:
            humidity_config = quality_config['humidity']
            invalid_humidity = df[
                (df['humidity'] < humidity_config['min_value']) | 
                (df['humidity'] > humidity_config['max_value'])
            ]
            if len(invalid_humidity) > 0:
                issues.append(f"Found {len(invalid_humidity)} humidity values outside valid range")
        
        # Pressure checks
        if 'pressure' in df.columns and 'pressure' in quality_config:
            pressure_config = quality_config['pressure']
            invalid_pressure = df[
                (df['pressure'] < pressure_config['min_value']) | 
                (df['pressure'] > pressure_config['max_value'])
            ]
            if len(invalid_pressure) > 0:
                issues.append(f"Found {len(invalid_pressure)} pressure values outside valid range")
        
        return issues
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        issues = validation_report.get('issues_found', [])
        
        if any('missing values' in issue for issue in issues):
            recommendations.append("Consider using interpolation or imputation for missing values")
        
        if any('outliers' in issue for issue in issues):
            recommendations.append("Review and potentially remove or cap outlier values")
        
        if any('date gaps' in issue for issue in issues):
            recommendations.append("Fill date gaps with interpolated or forward-filled values")
        
        if any('outside valid range' in issue for issue in issues):
            recommendations.append("Clean or remove values outside valid ranges")
        
        return recommendations
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values, outliers, and data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Convert data types
        df_clean = self._convert_data_types(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        type_config = self.config['processing']['type_conversion']
        
        # Convert datetime columns
        for col in type_config.get('datetime_columns', []):
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        # Convert numeric columns
        for col in type_config.get('numeric_columns', []):
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")
        
        # Convert categorical columns
        for col in type_config.get('categorical_columns', []):
            if col in df.columns:
                try:
                    df[col] = df[col].astype('category')
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to category: {e}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        missing_config = self.config['processing']['missing_values']
        strategy = missing_config['strategy']
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'interpolate':
            method = missing_config.get('interpolation_method', 'linear')
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].interpolate(method=method)
        elif strategy == 'mean':
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        method = self.config['processing']['validation']['outlier_method']
        threshold = self.config['processing']['validation']['outlier_threshold']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in df.columns and not df[col].empty:
                if method == 'iqr':
                    outlier_indices = self._detect_outliers_iqr(df[col])
                elif method == 'zscore':
                    outlier_indices = self._detect_outliers_zscore(df[col], threshold)
                else:
                    continue
                
                # Cap outliers instead of removing them
                if outlier_indices:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'temp') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing features for machine learning...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        target = df[target_column].copy()
        features_df = df.drop(columns=[target_column]).copy()
        
        # Handle datetime features
        if 'datetime' in features_df.columns:
            features_df = self._create_datetime_features(features_df)
        
        # Handle categorical features
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            features_df = self._encode_categorical_features(features_df, categorical_cols)
        
        # Scale numerical features
        numerical_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            features_df = self._scale_numerical_features(features_df, numerical_cols)
        
        logger.info(f"Feature preparation completed. Features shape: {features_df.shape}")
        return features_df, target
    
    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from datetime column."""
        if 'datetime' in df.columns:
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['dayofyear'] = df['datetime'].dt.dayofyear
            df['quarter'] = df['datetime'].dt.quarter
            
            # Cyclical encoding for periodic features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            
            # Remove original datetime column
            df = df.drop(columns=['datetime'])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        for col in categorical_cols:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features."""
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers['features'] = scaler
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                           output_dir: str = "data/processed/daily/"):
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Name of output file
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        
        # Save processing metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'records_processed': len(df),
            'features_count': len(df.columns),
            'processing_config': self.config,
            'scalers_used': list(self.scalers.keys()) if self.scalers else []
        }
        
        metadata_file = filepath.replace('.csv', '_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Processing metadata saved to {metadata_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Hanoi weather data")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="data/processed/daily/",
                       help="Output directory for processed data")
    parser.add_argument("--config", default="config/data_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--target", default="temp",
                       help="Target column name")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = WeatherDataProcessor(config_path=args.config)
        
        # Load data
        logger.info(f"Loading data from {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        # Validate data
        df, validation_report = processor.validate_data(df)
        logger.info(f"Validation completed with {len(validation_report['issues_found'])} issues found")
        
        # Clean data
        df_clean = processor.clean_data(df)
        
        # Prepare features
        features_df, target = processor.prepare_features(df_clean, target_column=args.target)
        
        # Combine features and target for saving
        processed_df = features_df.copy()
        processed_df[args.target] = target
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"hanoi_weather_processed_{timestamp}.csv"
        processor.save_processed_data(processed_df, output_filename, args.output_dir)
        
        logger.info("Data processing completed successfully!")
    
    except Exception as e:
        logger.error(f"Data processing failed with error: {e}")


if __name__ == "__main__":
    main()