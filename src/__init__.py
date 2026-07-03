"""
CareerGapAI - HR Analytics & Career Progression Analysis

A comprehensive machine learning solution for analyzing employee career progression,
identifying promotion stagnation, and detecting retention opportunities at scale.

Version: 1.0.0
"""

from .data_loader import DataLoader
from .eda import ExploratoryDataAnalysis
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .clustering import CareerPathClustering
from .analysis import CareerAnalysis

__all__ = ['DataLoader', 'ExploratoryDataAnalysis', 'FeatureEngineer', 'DataPreprocessor', 
           'CareerPathClustering', 'CareerAnalysis']

__version__ = '1.0.0'
__author__ = 'CareerGapAI Team'
