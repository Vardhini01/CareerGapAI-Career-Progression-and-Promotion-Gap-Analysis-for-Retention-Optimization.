"""
Data Loader Module - CareerGapAI

This module handles all data loading, validation, and initial preprocessing tasks.
It provides robust error handling, logging, and data quality checks.

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A comprehensive data loader class for handling HR analytics datasets.
    
    This class provides methods for:
    - Loading data from various file formats (CSV, Excel, Parquet)
    - Data validation and quality checks
    - Initial missing value handling
    - Data shape and type exploration
    
    Attributes:
        data_path (str): Path to the data directory
        df (pd.DataFrame): Loaded dataframe
        original_shape (Tuple): Shape of original dataframe
        data_info (Dict): Metadata about the dataset
    """
    
    def __init__(self, data_path: str = 'data'):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the data directory. Default is 'data'
        """
        self.data_path = Path(data_path)
        self.df = None
        self.original_shape = None
        self.data_info = {}
        logger.info(f"DataLoader initialized with data path: {self.data_path}")
    
    def load_data(self, filename: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            filename (str): Name of the file to load
            file_type (str): Type of file ('csv', 'excel', 'parquet'). Default is 'csv'
        
        Returns:
            pd.DataFrame: Loaded dataframe
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
            Exception: If data loading fails
        """
        file_path = self.data_path / filename
        
        # Validate file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        try:
            if file_type.lower() == 'csv':
                self.df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded CSV file: {filename}")
            
            elif file_type.lower() == 'excel':
                self.df = pd.read_excel(file_path)
                logger.info(f"Successfully loaded Excel file: {filename}")
            
            elif file_type.lower() == 'parquet':
                self.df = pd.read_parquet(file_path)
                logger.info(f"Successfully loaded Parquet file: {filename}")
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.original_shape = self.df.shape
            self._log_data_info()
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _log_data_info(self):
        """
        Log comprehensive information about the loaded dataset.
        """
        self.data_info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }
        
        logger.info(f"Dataset Shape: {self.data_info['shape']}")
        logger.info(f"Memory Usage: {self.data_info['memory_usage']:.2f} MB")
        logger.info(f"Total Missing Values: {self.df.isnull().sum().sum()}")
    
    def validate_required_columns(self, required_columns: list) -> bool:
        """
        Validate that all required columns exist in the dataset.
        
        Args:
            required_columns (list): List of column names that must exist
        
        Returns:
            bool: True if all columns exist, raises exception otherwise
        """
        missing_columns = set(required_columns) - set(self.df.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"All {len(required_columns)} required columns are present")
        return True
    
    def handle_missing_values(self, 
                             strategy: str = 'drop',
                             numeric_strategy: str = 'mean',
                             categorical_strategy: str = 'mode') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): 'drop' to remove rows/columns or 'fill' to impute
            numeric_strategy (str): Strategy for numeric columns ('mean', 'median', 'forward_fill')
            categorical_strategy (str): Strategy for categorical columns ('mode', 'forward_fill')
        
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        missing_before = self.df.isnull().sum().sum()
        logger.info(f"Missing values before handling: {missing_before}")
        
        if strategy == 'drop':
            # Drop columns with >50% missing values
            cols_to_drop = [col for col in self.df.columns 
                          if self.df[col].isnull().sum() / len(self.df) > 0.5]
            self.df.drop(columns=cols_to_drop, inplace=True)
            # Drop rows with any remaining missing values
            self.df.dropna(inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} columns and rows with missing values")
        
        elif strategy == 'fill':
            # Fill numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    if numeric_strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif numeric_strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif numeric_strategy == 'forward_fill':
                        self.df[col].fillna(method='ffill', inplace=True)
            
            # Fill categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    if categorical_strategy == 'mode':
                        mode_val = self.df[col].mode()[0]
                        self.df[col].fillna(mode_val, inplace=True)
                    elif categorical_strategy == 'forward_fill':
                        self.df[col].fillna(method='ffill', inplace=True)
            
            logger.info(f"Imputed missing values using {numeric_strategy}/{categorical_strategy}")
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return self.df
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistical summary of the dataset.
        
        Returns:
            Dict: Dictionary containing statistical information
        """
        stats = {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        logger.info(f"Dataset Statistics: {stats}")
        return stats
    
    def remove_duplicates(self, subset: Optional[list] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset (list, optional): Columns to consider for identifying duplicates
            keep (str): 'first', 'last', or False to remove all duplicates
        
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        duplicates_before = self.df.duplicated().sum()
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        duplicates_after = self.df.duplicated().sum()
        
        logger.info(f"Removed {duplicates_before - duplicates_after} duplicate rows")
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current dataframe.
        
        Returns:
            pd.DataFrame: The loaded and processed dataframe
        """
        if self.df is None:
            logger.warning("No data loaded. Please call load_data() first.")
            return None
        return self.df
    
    def save_processed_data(self, filename: str, file_type: str = 'csv') -> None:
        """
        Save processed data to file.
        
        Args:
            filename (str): Name of the file to save
            file_type (str): Format to save in ('csv', 'excel', 'parquet')
        """
        output_path = self.data_path / filename
        
        try:
            if file_type.lower() == 'csv':
                self.df.to_csv(output_path, index=False)
            elif file_type.lower() == 'excel':
                self.df.to_excel(output_path, index=False)
            elif file_type.lower() == 'parquet':
                self.df.to_parquet(output_path, index=False)
            
            logger.info(f"Data saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def display_data_summary(self) -> None:
        """
        Display a comprehensive summary of the dataset.
        """
        logger.info("=" * 80)
        logger.info("DATA SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"\nFirst few rows:\n{self.df.head()}")
        logger.info(f"\nData types:\n{self.df.dtypes}")
        logger.info(f"\nMissing values:\n{self.df.isnull().sum()}")
        logger.info(f"\nBasic statistics:\n{self.df.describe()}")
        logger.info("=" * 80)


if __name__ == "__main__":
    """
    Example usage of the DataLoader class.
    """
    # Initialize data loader
    loader = DataLoader(data_path='data')
    
    # Load data (replace 'sample.csv' with your actual filename)
    # df = loader.load_data('sample.csv', file_type='csv')
    
    # Validate required columns
    # required_cols = ['Age', 'Attrition', 'Department', 'JobRole', 'MonthlyIncome']
    # loader.validate_required_columns(required_cols)
    
    # Handle missing values
    # df = loader.handle_missing_values(strategy='fill', numeric_strategy='mean')
    
    # Get statistics
    # stats = loader.get_basic_statistics()
    
    # Display summary
    # loader.display_data_summary()
    
    print("DataLoader module ready for use.")
