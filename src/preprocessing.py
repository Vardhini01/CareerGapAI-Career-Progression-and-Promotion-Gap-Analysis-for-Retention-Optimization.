"""
Data Preprocessing Module - CareerGapAI

This module handles data cleaning, outlier removal, normalization, and transformation.
Designed specifically for HR analytics with career progression focus.

Key Responsibilities:
- Handle missing values and anomalies
- Remove extreme outliers (late-career edge cases)
- Normalize numerical career features
- Encode categorical fields (department, role, etc.)
- Data quality validation and reporting

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for HR analytics.
    
    This class handles all data preparation tasks including:
    - Outlier detection and removal
    - Missing value handling
    - Feature normalization and scaling
    - Categorical encoding
    - Data validation and quality checks
    
    Attributes:
        df (pd.DataFrame): The dataset to preprocess
        numeric_cols (list): List of numeric columns
        categorical_cols (list): List of categorical columns
        career_cols (list): List of career-specific columns
        transformers (Dict): Fitted transformers for reproducibility
        preprocessing_report (Dict): Log of all preprocessing operations
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataPreprocessor.
        
        Args:
            dataframe (pd.DataFrame): The dataset to preprocess
        """
        self.df = dataframe.copy()
        self.original_shape = self.df.shape
        self.original_columns = self.df.columns.tolist()
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Career-specific columns
        self.career_cols = [
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager', 'TrainingTimesLastYear', 'MonthlyIncome',
            'PerformanceRating', 'TotalWorkingYears'
        ]
        
        self.transformers = {}
        self.preprocessing_report = {
            'original_shape': self.original_shape,
            'operations': []
        }
        
        logger.info(f"DataPreprocessor initialized: {len(self.numeric_cols)} numeric, "
                   f"{len(self.categorical_cols)} categorical columns")
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            column (str): Name of column to analyze
            multiplier (float): IQR multiplier (1.5 = standard, 3.0 = extreme)
        
        Returns:
            np.ndarray: Boolean array of outlier indices
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return np.array([False] * len(self.df))
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        logger.info(f"IQR Outlier Detection - {column}: {outliers.sum()} outliers found "
                   f"(bounds: {lower_bound:.2f} to {upper_bound:.2f})")
        
        return outliers
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            column (str): Name of column to analyze
            threshold (float): Z-score threshold (3.0 = 99.7% of data)
        
        Returns:
            np.ndarray: Boolean array of outlier indices
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return np.array([False] * len(self.df))
        
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outliers = np.abs(stats.zscore(self.df[column].fillna(self.df[column].mean()))) > threshold
        
        logger.info(f"Z-score Outlier Detection - {column}: {outliers.sum()} outliers found "
                   f"(threshold: {threshold})")
        
        return outliers
    
    def remove_late_career_outliers(self, strategy: str = 'iqr') -> pd.DataFrame:
        """
        Remove extreme late-career outliers while preserving legitimate senior employees.
        
        Uses domain knowledge to handle edge cases in:
        - YearsAtCompany > reasonable tenure (e.g., >30 years)
        - YearsSinceLastPromotion > career length
        - MonthlyIncome with extreme disparities
        
        Args:
            strategy (str): 'iqr' for IQR-based or 'domain' for business rules
        
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        rows_removed = 0
        
        if strategy == 'domain':
            # Apply business logic rules for career data
            initial_len = len(self.df)
            
            # Rule 1: YearsSinceLastPromotion cannot exceed YearsAtCompany
            if 'YearsSinceLastPromotion' in self.df.columns and 'YearsAtCompany' in self.df.columns:
                invalid = self.df['YearsSinceLastPromotion'] > self.df['YearsAtCompany']
                self.df = self.df[~invalid]
                rows_removed += invalid.sum()
                logger.info(f"Removed {invalid.sum()} rows where "
                           f"YearsSinceLastPromotion > YearsAtCompany")
            
            # Rule 2: YearsInCurrentRole cannot exceed YearsAtCompany
            if 'YearsInCurrentRole' in self.df.columns and 'YearsAtCompany' in self.df.columns:
                invalid = self.df['YearsInCurrentRole'] > self.df['YearsAtCompany']
                self.df = self.df[~invalid]
                rows_removed += invalid.sum()
                logger.info(f"Removed {invalid.sum()} rows where "
                           f"YearsInCurrentRole > YearsAtCompany")
            
            # Rule 3: YearsWithCurrManager cannot exceed YearsAtCompany
            if 'YearsWithCurrManager' in self.df.columns and 'YearsAtCompany' in self.df.columns:
                invalid = self.df['YearsWithCurrManager'] > self.df['YearsAtCompany']
                self.df = self.df[~invalid]
                rows_removed += invalid.sum()
                logger.info(f"Removed {invalid.sum()} rows where "
                           f"YearsWithCurrManager > YearsAtCompany")
            
            # Rule 4: Remove extreme age outliers (< 18 or > 70)
            if 'Age' in self.df.columns:
                invalid = (self.df['Age'] < 18) | (self.df['Age'] > 70)
                self.df = self.df[~invalid]
                rows_removed += invalid.sum()
                logger.info(f"Removed {invalid.sum()} rows with invalid Age values")
        
        elif strategy == 'iqr':
            # Use IQR for career columns
            for col in self.career_cols:
                if col in self.df.columns:
                    outliers = self.detect_outliers_iqr(col, multiplier=3.0)
                    self.df = self.df[~outliers]
                    rows_removed += outliers.sum()
        
        self.preprocessing_report['operations'].append({
            'operation': 'remove_late_career_outliers',
            'strategy': strategy,
            'rows_removed': rows_removed,
            'remaining_rows': len(self.df)
        })
        
        logger.info(f"Late-career outlier removal: Removed {rows_removed} rows, "
                   f"remaining shape: {self.df.shape}")
        
        return self.df
    
    def handle_missing_values_advanced(self, 
                                      numeric_strategy: str = 'median',
                                      categorical_strategy: str = 'mode',
                                      missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        Advanced missing value handling with strategic imputation.
        
        Args:
            numeric_strategy (str): 'mean', 'median', 'forward_fill', 'knn'
            categorical_strategy (str): 'mode', 'forward_fill', 'unknown'
            missing_threshold (float): Columns with > threshold missing are dropped
        
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        missing_before = self.df.isnull().sum().sum()
        logger.info(f"Missing values before handling: {missing_before}")
        
        # Step 1: Drop columns with too many missing values
        cols_to_drop = [
            col for col in self.df.columns
            if self.df[col].isna().mean() > missing_threshold
        ]
        
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing: "
                       f"{cols_to_drop}")
        
        # Step 2: Impute numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if numeric_strategy == 'mean':
                    fill_value = self.df[col].mean()
                    self.df[col].fillna(fill_value, inplace=True)
                elif numeric_strategy == 'median':
                    fill_value = self.df[col].median()
                    self.df[col].fillna(fill_value, inplace=True)
                elif numeric_strategy == 'forward_fill':
                    self.df[col].fillna(method='ffill', inplace=True)
                    self.df[col].fillna(method='bfill', inplace=True)
        
        # Step 3: Impute categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                if categorical_strategy == 'mode':
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                elif categorical_strategy == 'unknown':
                    self.df[col].fillna('Unknown', inplace=True)
                elif categorical_strategy == 'forward_fill':
                    self.df[col].fillna(method='ffill', inplace=True)
                    self.df[col].fillna(method='bfill', inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        self.preprocessing_report['operations'].append({
            'operation': 'handle_missing_values',
            'strategy': f'numeric:{numeric_strategy}, categorical:{categorical_strategy}',
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'columns_dropped': len(cols_to_drop)
        })
        
        logger.info(f"Missing value handling: {missing_before} → {missing_after} missing values")
        
        return self.df
    
    def normalize_career_features(self, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize career-specific numerical features.
        
        Applies scaling to key career progression metrics to put them on
        comparable scales (typically 0-1 or standardized).
        
        Args:
            method (str): 'minmax' (0-1 range), 'standard' (mean=0, std=1), 'robust' (median-based)
        
        Returns:
            pd.DataFrame: Dataframe with normalized career features
        """
        # Select career features that exist in dataset
        features_to_normalize = [col for col in self.career_cols if col in self.df.columns]
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.error(f"Unknown scaling method: {method}")
            return self.df
        
        # Fit and transform
        self.df[features_to_normalize] = scaler.fit_transform(self.df[features_to_normalize])
        self.transformers[f'career_scaler_{method}'] = scaler
        
        self.preprocessing_report['operations'].append({
            'operation': 'normalize_career_features',
            'method': method,
            'features_normalized': len(features_to_normalize),
            'feature_list': features_to_normalize
        })
        
        logger.info(f"Normalized {len(features_to_normalize)} career features using {method} scaling")
        
        return self.df
    
    def encode_categorical_features(self, 
                                   columns: Optional[List[str]] = None,
                                   method: str = 'label',
                                   high_cardinality_threshold: int = 10) -> pd.DataFrame:
        """
        Encode categorical features intelligently.
        
        Strategy:
        - Low cardinality (< threshold): One-hot encoding
        - High cardinality: Label encoding
        
        Args:
            columns (List[str], optional): Specific columns to encode.
                                          If None, encodes all categorical columns.
            method (str): 'label', 'onehot', or 'auto' (intelligent selection)
            high_cardinality_threshold (int): Threshold for high-cardinality features
        
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = self.df.copy()
        
        # Determine columns to encode
        if columns is None:
            columns_to_encode = self.categorical_cols
        else:
            columns_to_encode = columns
        
        if method == 'auto':
            onehot_cols = []
            label_cols = []
            
            for col in columns_to_encode:
                unique_count = df_encoded[col].nunique()
                if unique_count <= high_cardinality_threshold:
                    onehot_cols.append(col)
                else:
                    label_cols.append(col)
            
            # One-hot encode low cardinality
            if onehot_cols:
                df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, 
                                           drop_first=True, dtype=int)
                logger.info(f"One-hot encoded {len(onehot_cols)} low-cardinality columns: {onehot_cols}")
            
            # Label encode high cardinality
            for col in label_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.transformers[f'label_encoder_{col}'] = le
            
            if label_cols:
                logger.info(f"Label encoded {len(label_cols)} high-cardinality columns: {label_cols}")
        
        elif method == 'label':
            for col in columns_to_encode:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.transformers[f'label_encoder_{col}'] = le
            
            logger.info(f"Label encoded {len(columns_to_encode)} categorical columns")
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode, 
                                       drop_first=True, dtype=int)
            logger.info(f"One-hot encoded {len(columns_to_encode)} categorical columns")
        
        self.preprocessing_report['operations'].append({
            'operation': 'encode_categorical_features',
            'method': method,
            'columns_encoded': len(columns_to_encode),
            'resulting_features': df_encoded.shape[1]
        })
        
        self.df = df_encoded
        return self.df
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset (List[str], optional): Columns to consider for duplicates
        
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        duplicates_before = self.df.duplicated().sum()
        self.df.drop_duplicates(subset=subset, keep='first', inplace=True)
        duplicates_after = self.df.duplicated().sum()
        
        removed = duplicates_before - duplicates_after
        
        self.preprocessing_report['operations'].append({
            'operation': 'remove_duplicates',
            'duplicates_removed': int(removed),
            'remaining_rows': len(self.df)
        })
        
        logger.info(f"Removed {removed} duplicate rows")
        
        return self.df
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Returns:
            Dict: Quality metrics and issues found
        """
        quality_report = {
            'shape': self.df.shape,
            'missing_values': int(self.df.isnull().sum().sum()),
            'missing_percentage': float((self.df.isnull().sum().sum() / 
                                       (self.df.shape[0] * self.df.shape[1])) * 100),
            'duplicates': int(self.df.duplicated().sum()),
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'issues': []
        }
        
        # Check for remaining issues
        if quality_report['missing_values'] > 0:
            quality_report['issues'].append(
                f"Found {quality_report['missing_values']} missing values"
            )
        
        if quality_report['duplicates'] > 0:
            quality_report['issues'].append(
                f"Found {quality_report['duplicates']} duplicate rows"
            )
        
        # Check numeric columns for unrealistic values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].min() < 0 and col not in ['MonthlyIncome', 'DailyRate', 'HourlyRate']:
                quality_report['issues'].append(
                    f"Column {col} has negative values"
                )
        
        logger.info(f"Data Quality Report: {quality_report['shape']}, "
                   f"Missing: {quality_report['missing_percentage']:.2f}%, "
                   f"Issues: {len(quality_report['issues'])}")
        
        return quality_report
    
    def get_preprocessing_summary(self) -> str:
        """
        Generate a comprehensive preprocessing summary.
        
        Returns:
            str: Formatted preprocessing summary
        """
        summary = []
        summary.append("=" * 80)
        summary.append("DATA PREPROCESSING SUMMARY - CareerGapAI")
        summary.append("=" * 80)
        
        summary.append(f"\nOriginal Shape: {self.preprocessing_report['original_shape']}")
        summary.append(f"Final Shape: {self.df.shape}")
        summary.append(f"Rows Removed: {self.preprocessing_report['original_shape'][0] - self.df.shape[0]}")
        summary.append(f"Rows Retained: {self.df.shape[0]}")
        
        summary.append(f"\nPreprocessing Operations ({len(self.preprocessing_report['operations'])}):")
        for i, op in enumerate(self.preprocessing_report['operations'], 1):
            summary.append(f"\n  {i}. {op['operation']}")
            for key, value in op.items():
                if key != 'operation':
                    summary.append(f"     - {key}: {value}")
        
        quality = self.validate_data_quality()
        summary.append(f"\nFinal Data Quality:")
        summary.append(f"  - Missing Values: {quality['missing_values']} ({quality['missing_percentage']:.2f}%)")
        summary.append(f"  - Duplicates: {quality['duplicates']}")
        summary.append(f"  - Total Columns: {len(quality['columns'])}")
        
        if quality['issues']:
            summary.append(f"\nRemaining Issues:")
            for issue in quality['issues']:
                summary.append(f"  ⚠ {issue}")
        else:
            summary.append(f"\n✓ No data quality issues detected")
        
        summary.append("\n" + "=" * 80)
        
        return "\n".join(summary)
    
    def apply_full_preprocessing_pipeline(self,
                                         remove_outliers: bool = True,
                                         handle_missing: bool = True,
                                         normalize: bool = True,
                                         encode_categorical: bool = True,
                                         remove_dups: bool = True) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline in optimal order.
        
        Pipeline order:
        1. Remove duplicates
        2. Remove outliers
        3. Handle missing values
        4. Encode categorical features
        5. Normalize numerical features
        
        Args:
            remove_outliers (bool): Remove outliers
            handle_missing (bool): Handle missing values
            normalize (bool): Normalize features
            encode_categorical (bool): Encode categorical features
            remove_dups (bool): Remove duplicates
        
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        logger.info("Starting full preprocessing pipeline...")
        
        if remove_dups:
            self.remove_duplicates()
        
        if remove_outliers:
            self.remove_late_career_outliers(strategy='domain')
        
        if handle_missing:
            self.handle_missing_values_advanced(numeric_strategy='median',
                                               categorical_strategy='mode')
        
        if encode_categorical:
            self.encode_categorical_features(method='auto')
        
        if normalize:
            # Re-select numeric columns after encoding
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            career_features = [col for col in self.career_cols if col in numeric_cols]
            if career_features:
                self.normalize_career_features(method='minmax')
        
        logger.info("Full preprocessing pipeline completed")
        
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the preprocessed dataframe.
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        return self.df
    
    def get_transformers(self) -> Dict:
        """
        Get fitted transformers for applying to new data.
        
        Returns:
            Dict: Dictionary of fitted transformers
        """
        return self.transformers


if __name__ == "__main__":
    """
    Example usage of the DataPreprocessor class.
    """
    # Load data
    # df = pd.read_csv('data/sample.csv')
    
    # Initialize preprocessor
    # preprocessor = DataPreprocessor(df)
    
    # Option 1: Apply full pipeline
    # preprocessed_df = preprocessor.apply_full_preprocessing_pipeline()
    
    # Option 2: Apply individual operations
    # preprocessor.remove_duplicates()
    # preprocessor.remove_late_career_outliers(strategy='domain')
    # preprocessor.handle_missing_values_advanced()
    # preprocessor.encode_categorical_features(method='auto')
    # preprocessor.normalize_career_features(method='minmax')
    
    # Get summary
    # summary = preprocessor.get_preprocessing_summary()
    # print(summary)
    
    print("DataPreprocessor module ready for use.")
