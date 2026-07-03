"""
Retention Risk Scoring Module - CareerGapAI

Calculates employee retention risk scores (0-100) using weighted metrics:
- YearsSinceLastPromotion (25%)
- YearsInCurrentRole (20%)
- JobSatisfaction (20%)
- TrainingTimesLastYear (15%)
- WorkLifeBalance (12%)
- YearsWithCurrManager (8%)

Risk Categories:
- Low Risk: 0-35
- Medium Risk: 35-65
- High Risk: 65-100

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RetentionRiskScorer:
    """Risk scoring engine for employee retention analysis."""
    
    # Risk weights (must sum to 1.0)
    WEIGHTS = {
        'YearsSinceLastPromotion': 0.25,
        'YearsInCurrentRole': 0.20,
        'JobSatisfaction': 0.20,
        'TrainingTimesLastYear': 0.15,
        'WorkLifeBalance': 0.12,
        'YearsWithCurrManager': 0.08
    }
    
    # Risk thresholds
    LOW_RISK_THRESHOLD = 35
    MEDIUM_RISK_THRESHOLD = 65
    
    # Risk colors
    RISK_COLORS = {
        'Low Risk': '#4CAF50',      # Green
        'Medium Risk': '#FF9800',   # Orange
        'High Risk': '#F44336'      # Red
    }
    
    @staticmethod
    def normalize_metric(value: float, metric_name: str, max_val: Optional[float] = None) -> float:
        """
        Normalize metric to 0-1 scale (higher value = higher risk).
        
        Args:
            value: Raw metric value
            metric_name: Name of the metric
            max_val: Maximum value for normalization (auto-calculated if None)
            
        Returns:
            Normalized value (0-1), higher = riskier
        """
        if pd.isna(value):
            return 0.5  # Neutral value for missing data
        
        # For satisfaction/balance metrics (1-4 scale): invert (4=low risk, 1=high risk)
        if metric_name in ['JobSatisfaction', 'WorkLifeBalance']:
            # Scale 1-4 to 0-1, inverted: 4->0 (safe), 1->1 (risky)
            return max(0, min(1, (4 - value) / 3))
        
        # For training times: more training = lower risk
        if metric_name == 'TrainingTimesLastYear':
            # Invert: 0 trainings = high risk, 5+ = low risk
            return max(0, min(1, 1 - (value / 5)))
        
        # For tenure metrics: normalize to 0-1 scale
        if metric_name == 'YearsSinceLastPromotion':
            return max(0, min(1, value / 10))  # Assume 10 years as max realistic
        
        if metric_name == 'YearsInCurrentRole':
            return max(0, min(1, value / 8))   # Assume 8 years as max realistic
        
        if metric_name == 'YearsWithCurrManager':
            return max(0, min(1, value / 8))   # Assume 8 years as max realistic
        
        return 0.5
    
    @staticmethod
    def calculate_risk_score(employee_row: pd.Series) -> float:
        """
        Calculate retention risk score for an employee (0-100).
        
        Args:
            employee_row: Pandas Series with employee data
            
        Returns:
            Risk score (0-100), where 100 is highest risk
        """
        score = 0.0
        
        for metric, weight in RetentionRiskScorer.WEIGHTS.items():
            if metric in employee_row.index:
                value = employee_row[metric]
                normalized = RetentionRiskScorer.normalize_metric(value, metric)
                score += normalized * weight
        
        return score * 100
    
    @staticmethod
    def categorize_risk(risk_score: float) -> str:
        """
        Categorize risk score into risk level.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk category: 'Low Risk', 'Medium Risk', or 'High Risk'
        """
        if risk_score < RetentionRiskScorer.LOW_RISK_THRESHOLD:
            return 'Low Risk'
        elif risk_score < RetentionRiskScorer.MEDIUM_RISK_THRESHOLD:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    @staticmethod
    def get_risk_color(risk_score: float) -> str:
        """Get color for risk visualization."""
        category = RetentionRiskScorer.categorize_risk(risk_score)
        return RetentionRiskScorer.RISK_COLORS.get(category, '#999999')
    
    @staticmethod
    def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk score columns to dataframe.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dataframe with added RiskScore, RiskCategory, RiskColor columns
        """
        df = df.copy()
        
        # Calculate scores
        df['RiskScore'] = df.apply(RetentionRiskScorer.calculate_risk_score, axis=1)
        df['RiskCategory'] = df['RiskScore'].apply(RetentionRiskScorer.categorize_risk)
        df['RiskColor'] = df['RiskScore'].apply(RetentionRiskScorer.get_risk_color)
        
        return df
    
    @staticmethod
    def get_risk_summary(df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics for risk scores.
        
        Args:
            df: Dataframe with RiskScore column
            
        Returns:
            Dictionary with risk summary
        """
        if 'RiskScore' not in df.columns:
            df = RetentionRiskScorer.add_risk_scores(df)
        
        total = len(df)
        low_risk = (df['RiskScore'] < 35).sum()
        medium_risk = ((df['RiskScore'] >= 35) & (df['RiskScore'] < 65)).sum()
        high_risk = (df['RiskScore'] >= 65).sum()
        
        return {
            'total_employees': total,
            'low_risk_count': low_risk,
            'medium_risk_count': medium_risk,
            'high_risk_count': high_risk,
            'low_risk_pct': (low_risk / total * 100) if total > 0 else 0,
            'medium_risk_pct': (medium_risk / total * 100) if total > 0 else 0,
            'high_risk_pct': (high_risk / total * 100) if total > 0 else 0,
            'avg_risk_score': df['RiskScore'].mean(),
            'median_risk_score': df['RiskScore'].median(),
            'max_risk_score': df['RiskScore'].max(),
            'min_risk_score': df['RiskScore'].min()
        }
    
    @staticmethod
    def get_top_risk_employees(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get top N highest-risk employees.
        
        Args:
            df: Employee dataframe
            n: Number of top employees to return
            
        Returns:
            Dataframe of top risk employees sorted by RiskScore
        """
        if 'RiskScore' not in df.columns:
            df = RetentionRiskScorer.add_risk_scores(df)
        
        cols = ['EmployeeID', 'Department', 'JobRole', 'RiskScore', 'RiskCategory',
                'YearsSinceLastPromotion', 'YearsInCurrentRole', 'JobSatisfaction',
                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsWithCurrManager']
        
        available_cols = [col for col in cols if col in df.columns]
        
        return df.nlargest(n, 'RiskScore')[available_cols]
    
    @staticmethod
    def get_department_risk_analysis(df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze risk by department.
        
        Args:
            df: Employee dataframe with RiskScore column
            
        Returns:
            Dictionary with per-department risk statistics
        """
        if 'RiskScore' not in df.columns:
            df = RetentionRiskScorer.add_risk_scores(df)
        
        dept_analysis = {}
        
        for dept in df['Department'].unique():
            dept_data = df[df['Department'] == dept]
            
            high_risk_count = (dept_data['RiskScore'] >= 65).sum()
            avg_risk = dept_data['RiskScore'].mean()
            
            dept_analysis[dept] = {
                'employee_count': len(dept_data),
                'avg_risk_score': avg_risk,
                'high_risk_count': high_risk_count,
                'high_risk_pct': (high_risk_count / len(dept_data) * 100) if len(dept_data) > 0 else 0,
                'median_risk': dept_data['RiskScore'].median(),
                'max_risk': dept_data['RiskScore'].max(),
                'min_risk': dept_data['RiskScore'].min()
            }
        
        return dept_analysis
    
    @staticmethod
    def get_risk_distribution_bins(df: pd.DataFrame, n_bins: int = 10) -> Tuple[List, List]:
        """
        Get distribution of risk scores in bins.
        
        Args:
            df: Employee dataframe
            n_bins: Number of bins
            
        Returns:
            Tuple of (bin_edges, counts)
        """
        if 'RiskScore' not in df.columns:
            df = RetentionRiskScorer.add_risk_scores(df)
        
        counts, bins = np.histogram(df['RiskScore'], bins=n_bins, range=(0, 100))
        
        return bins.tolist(), counts.tolist()
