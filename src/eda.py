"""
Exploratory Data Analysis Module - CareerGapAI

This module provides comprehensive exploratory data analysis functions for HR analytics.
It includes univariate, bivariate, and multivariate analysis functions with visualization support.

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ExploratoryDataAnalysis:
    """
    Comprehensive EDA toolkit for HR analytics dataset analysis.
    
    This class provides methods for:
    - Univariate analysis (distributions, summaries)
    - Bivariate analysis (correlations, relationships)
    - Multivariate analysis (patterns, groupings)
    - Data quality assessment
    - Visual reporting
    
    Attributes:
        df (pd.DataFrame): The dataset to analyze
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        target_col (str): Target variable column name (if applicable)
    """
    
    def __init__(self, dataframe: pd.DataFrame, target_col: Optional[str] = None):
        """
        Initialize the EDA class.
        
        Args:
            dataframe (pd.DataFrame): The dataset to analyze
            target_col (str, optional): Name of the target variable
        """
        self.df = dataframe.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.target_col = target_col
        
        logger.info(f"EDA initialized: {len(self.numeric_cols)} numeric, "
                   f"{len(self.categorical_cols)} categorical columns")
    
    def get_data_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile.
        
        Returns:
            Dict: Complete data profile information
        """
        profile = {
            'shape': self.df.shape,
            'total_missing': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / 
                                 (self.df.shape[0] * self.df.shape[1])) * 100,
            'duplicates': self.df.duplicated().sum(),
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'data_types': self.df.dtypes.value_counts().to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Data Profile: {profile['shape'][0]} rows, "
                   f"{profile['shape'][1]} columns")
        return profile
    
    def univariate_numeric_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform univariate analysis on numeric columns.
        
        Args:
            column (str): Name of numeric column
        
        Returns:
            Dict: Statistical summary of the column
        """
        if column not in self.numeric_cols:
            logger.warning(f"Column {column} is not numeric")
            return None
        
        analysis = {
            'column': column,
            'count': self.df[column].count(),
            'missing': self.df[column].isnull().sum(),
            'missing_percentage': (self.df[column].isnull().sum() / len(self.df)) * 100,
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'std': self.df[column].std(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'q1': self.df[column].quantile(0.25),
            'q3': self.df[column].quantile(0.75),
            'iqr': self.df[column].quantile(0.75) - self.df[column].quantile(0.25),
            'skewness': self.df[column].skew(),
            'kurtosis': self.df[column].kurtosis()
        }
        
        logger.info(f"Numeric Analysis - {column}: Mean={analysis['mean']:.2f}, "
                   f"Std={analysis['std']:.2f}")
        return analysis
    
    def univariate_categorical_analysis(self, column: str) -> Dict[str, Any]:
        """
        Perform univariate analysis on categorical columns.
        
        Args:
            column (str): Name of categorical column
        
        Returns:
            Dict: Summary of category distribution
        """
        if column not in self.categorical_cols:
            logger.warning(f"Column {column} is not categorical")
            return None
        
        value_counts = self.df[column].value_counts()
        
        analysis = {
            'column': column,
            'unique_values': self.df[column].nunique(),
            'most_frequent': value_counts.index[0],
            'most_frequent_count': value_counts.values[0],
            'least_frequent': value_counts.index[-1],
            'least_frequent_count': value_counts.values[-1],
            'missing': self.df[column].isnull().sum(),
            'missing_percentage': (self.df[column].isnull().sum() / len(self.df)) * 100,
            'value_distribution': value_counts.to_dict()
        }
        
        logger.info(f"Categorical Analysis - {column}: {analysis['unique_values']} unique values")
        return analysis
    
    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for numeric columns.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if len(self.numeric_cols) == 0:
            logger.warning("No numeric columns found for correlation analysis")
            return None
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        logger.info(f"Computed {method} correlation matrix: {corr_matrix.shape}")
        return corr_matrix
    
    def get_high_correlations(self, threshold: float = 0.7, 
                            method: str = 'pearson') -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated numeric column pairs.
        
        Args:
            threshold (float): Correlation threshold (default 0.7)
            method (str): Correlation method
        
        Returns:
            List: Tuples of (col1, col2, correlation_value)
        """
        corr_matrix = self.correlation_analysis(method=method)
        
        if corr_matrix is None:
            return []
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        logger.info(f"Found {len(high_corr)} highly correlated pairs (threshold={threshold})")
        return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)
    
    def target_analysis(self, target_col: str) -> Dict[str, Any]:
        """
        Analyze target variable distribution and characteristics.
        
        Args:
            target_col (str): Name of target column
        
        Returns:
            Dict: Target variable analysis
        """
        if target_col not in self.df.columns:
            logger.error(f"Target column {target_col} not found")
            return None
        
        if target_col in self.numeric_cols:
            analysis = self.univariate_numeric_analysis(target_col)
        else:
            analysis = self.univariate_categorical_analysis(target_col)
        
        logger.info(f"Target Analysis - {target_col}")
        return analysis
    
    def feature_vs_target_analysis(self, feature_col: str, 
                                   target_col: str) -> Dict[str, Any]:
        """
        Analyze relationship between a feature and target variable.
        
        Args:
            feature_col (str): Name of feature column
            target_col (str): Name of target column
        
        Returns:
            Dict: Analysis of feature-target relationship
        """
        analysis = {'feature': feature_col, 'target': target_col}
        
        if feature_col in self.numeric_cols and target_col in self.numeric_cols:
            # Both numeric: correlation
            corr = self.df[feature_col].corr(self.df[target_col])
            analysis['correlation'] = corr
            analysis['type'] = 'numeric_vs_numeric'
        
        elif feature_col in self.categorical_cols and target_col in self.numeric_cols:
            # Categorical feature vs numeric target
            group_stats = self.df.groupby(feature_col)[target_col].agg(['mean', 'std', 'count'])
            analysis['group_statistics'] = group_stats.to_dict()
            analysis['type'] = 'categorical_vs_numeric'
        
        elif feature_col in self.numeric_cols and target_col in self.categorical_cols:
            # Numeric feature vs categorical target
            group_stats = self.df.groupby(target_col)[feature_col].agg(['mean', 'std', 'count'])
            analysis['group_statistics'] = group_stats.to_dict()
            analysis['type'] = 'numeric_vs_categorical'
        
        else:
            # Both categorical: chi-square test
            crosstab = pd.crosstab(self.df[feature_col], self.df[target_col])
            chi2, p_value = stats.chi2_contingency(crosstab)[:2]
            analysis['chi2_statistic'] = chi2
            analysis['p_value'] = p_value
            analysis['type'] = 'categorical_vs_categorical'
        
        logger.info(f"Feature-Target Analysis: {feature_col} vs {target_col}")
        return analysis
    
    def outlier_detection(self, column: str, method: str = 'iqr',
                         threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Args:
            column (str): Name of numeric column
            method (str): 'iqr' or 'zscore'
            threshold (float): IQR multiplier or z-score threshold
        
        Returns:
            Dict: Outlier information
        """
        if column not in self.numeric_cols:
            logger.warning(f"Column {column} is not numeric")
            return None
        
        outlier_info = {'column': column, 'method': method}
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | 
                             (self.df[column] > upper_bound)]
            outlier_info['lower_bound'] = lower_bound
            outlier_info['upper_bound'] = upper_bound
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            outliers = self.df[np.abs(stats.zscore(self.df[column])) > threshold]
            outlier_info['zscore_threshold'] = threshold
        
        outlier_info['outlier_count'] = len(outliers)
        outlier_info['outlier_percentage'] = (len(outliers) / len(self.df)) * 100
        outlier_info['outlier_indices'] = outliers.index.tolist()
        
        logger.info(f"Outlier Detection ({method}) - {column}: "
                   f"{outlier_info['outlier_count']} outliers found")
        return outlier_info
    
    def missing_value_pattern(self) -> pd.DataFrame:
        """
        Analyze missing value patterns in the dataset.
        
        Returns:
            pd.DataFrame: Missing value summary
        """
        missing_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100)
        }).sort_values('Missing_Percentage', ascending=False)
        
        logger.info(f"Missing Value Analysis: {missing_summary['Missing_Count'].sum()} total missing")
        return missing_summary[missing_summary['Missing_Count'] > 0]
    
    def plot_numeric_distribution(self, column: str, bins: int = 30,
                                 figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot distribution of numeric column.
        
        Args:
            column (str): Name of numeric column
            bins (int): Number of bins for histogram
            figsize (Tuple): Figure size
        """
        if column not in self.numeric_cols:
            logger.warning(f"Column {column} is not numeric")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Distribution of {column}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        axes[1].boxplot(self.df[column].dropna())
        axes[1].set_title(f'Box Plot of {column}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel(column)
        
        plt.tight_layout()
        logger.info(f"Plotted distribution for {column}")
        return fig
    
    def plot_categorical_distribution(self, column: str,
                                     figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot distribution of categorical column.
        
        Args:
            column (str): Name of categorical column
            figsize (Tuple): Figure size
        """
        if column not in self.categorical_cols:
            logger.warning(f"Column {column} is not categorical")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        value_counts = self.df[column].value_counts()
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribution of {column}', fontsize=12, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        logger.info(f"Plotted distribution for {column}")
        return fig
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot correlation heatmap of numeric columns.
        
        Args:
            figsize (Tuple): Figure size
        """
        corr_matrix = self.correlation_analysis()
        
        if corr_matrix is None:
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Plotted correlation heatmap")
        return fig
    
    def generate_eda_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive EDA report.
        
        Args:
            output_file (str, optional): Path to save report
        
        Returns:
            str: EDA report text
        """
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT - CareerGapAI")
        report.append("=" * 80)
        
        # Data Profile
        profile = self.get_data_profile()
        report.append(f"\n1. DATA PROFILE")
        report.append(f"   Shape: {profile['shape']}")
        report.append(f"   Total Missing: {profile['total_missing']} ({profile['missing_percentage']:.2f}%)")
        report.append(f"   Duplicates: {profile['duplicates']}")
        report.append(f"   Memory Usage: {profile['memory_usage_mb']:.2f} MB")
        
        # Numeric columns summary
        report.append(f"\n2. NUMERIC COLUMNS SUMMARY")
        for col in self.numeric_cols[:5]:  # First 5 numeric columns
            analysis = self.univariate_numeric_analysis(col)
            report.append(f"   {col}: Mean={analysis['mean']:.2f}, "
                        f"Std={analysis['std']:.2f}, "
                        f"Min={analysis['min']:.2f}, Max={analysis['max']:.2f}")
        
        # High correlations
        high_corr = self.get_high_correlations(threshold=0.7)
        report.append(f"\n3. HIGH CORRELATIONS (threshold >= 0.7)")
        for col1, col2, corr in high_corr[:5]:
            report.append(f"   {col1} <-> {col2}: {corr:.3f}")
        
        # Missing values
        report.append(f"\n4. MISSING VALUE ANALYSIS")
        missing = self.missing_value_pattern()
        for _, row in missing.iterrows():
            report.append(f"   {row['Column']}: {row['Missing_Count']} "
                        f"({row['Missing_Percentage']:.2f}%)")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"EDA report saved to {output_file}")
        
        return report_text


if __name__ == "__main__":
    """
    Example usage of the ExploratoryDataAnalysis class.
    """
    # Load sample data (replace with your actual data)
    # df = pd.read_csv('data/sample.csv')
    
    # Initialize EDA
    # eda = ExploratoryDataAnalysis(df, target_col='Attrition')
    
    # Get data profile
    # profile = eda.get_data_profile()
    
    # Numeric analysis
    # numeric_analysis = eda.univariate_numeric_analysis('Age')
    
    # Correlation analysis
    # high_corr = eda.get_high_correlations(threshold=0.7)
    
    # Generate report
    # report = eda.generate_eda_report(output_file='outputs/eda_report.txt')
    
    print("EDA module ready for use.")
