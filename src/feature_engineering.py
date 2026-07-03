"""
Feature Engineering Module - CareerGapAI

This module handles creation of engineered career metrics and features for the HR analytics project.
It creates domain-specific features that capture career progression patterns and stagnation signals.

Main Features:
- PromotionGapRatio: Measures promotion delay relative to tenure
- RoleStagnationIndex: Identifies roles with limited advancement opportunities
- TrainingIntensityScore: Quantifies training investment and skill development
- ManagerStabilityIndicator: Measures managerial relationship stability

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for HR analytics.
    
    This class creates domain-specific career metrics and transforms raw HR data
    into meaningful features for machine learning models.
    
    Attributes:
        df (pd.DataFrame): The dataset for feature engineering
        engineered_features (Dict): Dictionary of created features
        feature_scalers (Dict): Dictionary of fitted scalers for reproducibility
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            dataframe (pd.DataFrame): The dataset for feature engineering
        """
        self.df = dataframe.copy()
        self.engineered_features = {}
        self.feature_scalers = {}
        self.label_encoders = {}
        
        logger.info(f"FeatureEngineer initialized with dataset shape: {self.df.shape}")
    
    def create_promotion_gap_ratio(self) -> pd.Series:
        """
        Calculate PromotionGapRatio: Ratio of years since last promotion to tenure.
        
        This metric identifies employees who have been stagnant in their role.
        High values indicate long periods without promotion.
        
        Formula: YearsSinceLastPromotion / (YearsAtCompany + 1)
        
        Returns:
            pd.Series: Promotion gap ratio for each employee
        """
        if 'YearsSinceLastPromotion' not in self.df.columns:
            logger.warning("'YearsSinceLastPromotion' column not found")
            return None
        
        if 'YearsAtCompany' not in self.df.columns:
            logger.warning("'YearsAtCompany' column not found")
            return None
        
        # Add 1 to YearsAtCompany to avoid division by zero
        promotion_gap = (self.df['YearsSinceLastPromotion'] / 
                        (self.df['YearsAtCompany'] + 1))
        
        # Cap the ratio at 1 (100% of tenure without promotion)
        promotion_gap = promotion_gap.clip(upper=1.0)
        
        self.engineered_features['PromotionGapRatio'] = promotion_gap
        logger.info(f"Created PromotionGapRatio: Min={promotion_gap.min():.3f}, "
                   f"Max={promotion_gap.max():.3f}, Mean={promotion_gap.mean():.3f}")
        
        return promotion_gap
    
    def create_role_stagnation_index(self) -> pd.Series:
        """
        Calculate RoleStagnationIndex: Measures how long an employee has been in current role.
        
        This identifies employees stuck in the same role with no advancement.
        Combines YearsInCurrentRole and PromotionGapRatio.
        
        Returns:
            pd.Series: Role stagnation index for each employee (0-1 scale)
        """
        if 'YearsInCurrentRole' not in self.df.columns:
            logger.warning("'YearsInCurrentRole' column not found")
            return None
        
        # Normalize YearsInCurrentRole to 0-1 scale
        max_years = self.df['YearsInCurrentRole'].max()
        role_tenure_normalized = self.df['YearsInCurrentRole'] / (max_years + 1)
        
        # Use PromotionGapRatio if available, otherwise create it
        if 'PromotionGapRatio' not in self.engineered_features:
            self.create_promotion_gap_ratio()
        
        promotion_gap = self.engineered_features['PromotionGapRatio']
        
        # Combine both factors (50% role tenure, 50% promotion gap)
        stagnation_index = 0.5 * role_tenure_normalized + 0.5 * promotion_gap
        
        # Clip to 0-1 range
        stagnation_index = stagnation_index.clip(0, 1)
        
        self.engineered_features['RoleStagnationIndex'] = stagnation_index
        logger.info(f"Created RoleStagnationIndex: Min={stagnation_index.min():.3f}, "
                   f"Max={stagnation_index.max():.3f}, Mean={stagnation_index.mean():.3f}")
        
        return stagnation_index
    
    def create_training_intensity_score(self) -> pd.Series:
        """
        Calculate TrainingIntensityScore: Measures training investment and skill development.
        
        Combines:
        - Annual training frequency (TrainingTimesLastYear)
        - Training normalized by tenure
        
        Formula: (TrainingTimesLastYear / max_training) * 0.6 + 
                 (TrainingTimesLastYear / YearsAtCompany) * 0.4
        
        Returns:
            pd.Series: Training intensity score for each employee (0-1 scale)
        """
        if 'TrainingTimesLastYear' not in self.df.columns:
            logger.warning("'TrainingTimesLastYear' column not found")
            return None
        
        if 'YearsAtCompany' not in self.df.columns:
            logger.warning("'YearsAtCompany' column not found")
            return None
        
        # Component 1: Normalized annual training
        max_training = self.df['TrainingTimesLastYear'].max()
        annual_training_normalized = self.df['TrainingTimesLastYear'] / (max_training + 1)
        
        # Component 2: Training per year of service
        training_per_year = self.df['TrainingTimesLastYear'] / (self.df['YearsAtCompany'] + 1)
        max_training_per_year = training_per_year.max()
        training_per_year_normalized = training_per_year / (max_training_per_year + 1)
        
        # Combine components
        training_intensity = (0.6 * annual_training_normalized + 
                            0.4 * training_per_year_normalized)
        
        # Clip to 0-1 range
        training_intensity = training_intensity.clip(0, 1)
        
        self.engineered_features['TrainingIntensityScore'] = training_intensity
        logger.info(f"Created TrainingIntensityScore: Min={training_intensity.min():.3f}, "
                   f"Max={training_intensity.max():.3f}, Mean={training_intensity.mean():.3f}")
        
        return training_intensity
    
    def create_manager_stability_indicator(self) -> pd.Series:
        """
        Calculate ManagerStabilityIndicator: Measures managerial relationship stability.
        
        Identifies employees with stable manager relationships.
        High values indicate consistent reporting relationships.
        
        Formula: YearsWithCurrManager / YearsAtCompany
        
        Returns:
            pd.Series: Manager stability indicator for each employee (0-1 scale)
        """
        if 'YearsWithCurrManager' not in self.df.columns:
            logger.warning("'YearsWithCurrManager' column not found")
            return None
        
        if 'YearsAtCompany' not in self.df.columns:
            logger.warning("'YearsAtCompany' column not found")
            return None
        
        # Calculate ratio of years with current manager to total tenure
        manager_stability = (self.df['YearsWithCurrManager'] / 
                            (self.df['YearsAtCompany'] + 1))
        
        # Clip to 0-1 range (some might be > 1 due to data quality)
        manager_stability = manager_stability.clip(0, 1)
        
        self.engineered_features['ManagerStabilityIndicator'] = manager_stability
        logger.info(f"Created ManagerStabilityIndicator: Min={manager_stability.min():.3f}, "
                   f"Max={manager_stability.max():.3f}, Mean={manager_stability.mean():.3f}")
        
        return manager_stability
    
    def create_all_career_metrics(self) -> pd.DataFrame:
        """
        Create all career metrics in one call.
        
        Returns:
            pd.DataFrame: Dataframe with all engineered career metrics
        """
        logger.info("Creating all career metrics...")
        
        self.create_promotion_gap_ratio()
        self.create_role_stagnation_index()
        self.create_training_intensity_score()
        self.create_manager_stability_indicator()
        
        # Combine all engineered features
        engineered_df = pd.DataFrame(self.engineered_features)
        logger.info(f"Created {len(engineered_df.columns)} career metrics")
        
        return engineered_df
    
    def encode_categorical_variables(self, columns: Optional[List[str]] = None,
                                    method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            columns (List[str], optional): Specific columns to encode. 
                                          If None, encodes all categorical columns.
            method (str): 'label' for label encoding, 'onehot' for one-hot encoding
        
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = self.df.copy()
        
        # Determine columns to encode
        if columns is None:
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        else:
            categorical_cols = columns
        
        if method == 'label':
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Label encoded column: {col}")
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                       drop_first=True)
            logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns")
        
        return df_encoded
    
    def scale_numeric_features(self, columns: Optional[List[str]] = None,
                              method: str = 'minmax') -> pd.DataFrame:
        """
        Scale numeric features for machine learning.
        
        Args:
            columns (List[str], optional): Specific columns to scale.
                                          If None, scales all numeric columns.
            method (str): 'minmax' for MinMaxScaler, 'standard' for StandardScaler
        
        Returns:
            pd.DataFrame: Dataframe with scaled numeric variables
        """
        df_scaled = self.df.copy()
        
        # Determine columns to scale
        if columns is None:
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            self.feature_scalers['minmax'] = scaler
        elif method == 'standard':
            scaler = StandardScaler()
            self.feature_scalers['standard'] = scaler
        else:
            logger.error(f"Unknown scaling method: {method}")
            return df_scaled
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        logger.info(f"Scaled {len(numeric_cols)} numeric columns using {method} scaling")
        
        return df_scaled
    
    def create_age_groups(self, column: str = 'Age',
                         bins: list = [20, 30, 40, 50, 60, 70]) -> pd.Series:
        """
        Create age group categories.
        
        Args:
            column (str): Name of age column
            bins (list): Age bin boundaries
        
        Returns:
            pd.Series: Age group categories
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return None
        
        labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
        age_groups = pd.cut(self.df[column], bins=bins, labels=labels, right=False)
        
        self.engineered_features['AgeGroup'] = age_groups
        logger.info(f"Created age groups with {len(labels)} categories")
        
        return age_groups
    
    def create_income_percentile(self, column: str = 'MonthlyIncome') -> pd.Series:
        """
        Create income percentile rank.
        
        Args:
            column (str): Name of income column
        
        Returns:
            pd.Series: Income percentile rank (0-100)
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return None
        
        income_percentile = self.df[column].rank(pct=True) * 100
        
        self.engineered_features['IncomePercentile'] = income_percentile
        logger.info(f"Created income percentile rank")
        
        return income_percentile
    
    def create_performance_score(self, performance_col: str = 'PerformanceRating',
                                training_col: str = 'TrainingTimesLastYear') -> pd.Series:
        """
        Create composite performance score.
        
        Combines performance rating with training investment.
        
        Args:
            performance_col (str): Name of performance rating column
            training_col (str): Name of training frequency column
        
        Returns:
            pd.Series: Composite performance score (0-1 scale)
        """
        if performance_col not in self.df.columns or training_col not in self.df.columns:
            logger.warning(f"Required columns not found")
            return None
        
        # Normalize performance rating
        perf_norm = self.df[performance_col] / self.df[performance_col].max()
        
        # Normalize training
        train_norm = self.df[training_col] / (self.df[training_col].max() + 1)
        
        # Combine (60% performance, 40% training)
        performance_score = 0.6 * perf_norm + 0.4 * train_norm
        
        self.engineered_features['CompositePerformanceScore'] = performance_score
        logger.info(f"Created composite performance score")
        
        return performance_score
    
    def get_engineered_features_dataframe(self) -> pd.DataFrame:
        """
        Get all engineered features as a dataframe.
        
        Returns:
            pd.DataFrame: All engineered features
        """
        return pd.DataFrame(self.engineered_features)
    
    def add_engineered_features_to_original(self) -> pd.DataFrame:
        """
        Add engineered features to the original dataframe without creating duplicates.
        
        Returns:
            pd.DataFrame: Original dataframe with engineered features appended
        """
        result_df = self.df.copy()
        engineered_df = self.get_engineered_features_dataframe()

        for feature_name in engineered_df.columns:
            if feature_name in result_df.columns:
                result_df[feature_name] = engineered_df[feature_name]
            else:
                result_df[feature_name] = engineered_df[feature_name]

        logger.info(f"Added {len(engineered_df.columns)} engineered features. "
                   f"Result shape: {result_df.shape}")

        return result_df
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary statistics of engineered features.
        
        Returns:
            Dict: Summary statistics for all engineered features
        """
        engineered_df = self.get_engineered_features_dataframe()
        
        summary = {}
        for col in engineered_df.columns:
            summary[col] = {
                'dtype': str(engineered_df[col].dtype),
                'min': float(engineered_df[col].min()),
                'max': float(engineered_df[col].max()),
                'mean': float(engineered_df[col].mean()),
                'std': float(engineered_df[col].std()),
                'missing': int(engineered_df[col].isnull().sum())
            }
        
        logger.info(f"Generated summary for {len(summary)} engineered features")
        return summary
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all engineered features and return dataframe with all features.
        
        This is a convenience method that creates all career metrics and adds them
        to the original dataframe in one call.
        
        Returns:
            pd.DataFrame: Original dataframe with all engineered features added
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Create all career metrics
        career_metrics = self.create_all_career_metrics()
        logger.info(f"Created {len(career_metrics.columns)} career metrics")
        
        # Add engineered features to original
        result_df = self.add_engineered_features_to_original()
        logger.info(f"Feature engineering complete. Final shape: {result_df.shape}")
        
        return result_df


if __name__ == "__main__":
    """
    Example usage of the FeatureEngineer class.
    """
    # Load sample data (replace with your actual data)
    # df = pd.read_csv('data/sample.csv')
    
    # Initialize feature engineer
    # engineer = FeatureEngineer(df)
    
    # Create all career metrics
    # career_metrics = engineer.create_all_career_metrics()
    
    # Add engineered features to original dataframe
    # df_with_features = engineer.add_engineered_features_to_original()
    
    # Get feature summary
    # feature_summary = engineer.get_feature_summary()
    
    print("FeatureEngineer module ready for use.")
