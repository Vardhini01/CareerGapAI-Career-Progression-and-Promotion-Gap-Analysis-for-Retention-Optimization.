"""
Analysis Module - CareerGapAI

This module performs in-depth analysis of career patterns, risk scoring, and
identification of retention opportunities.

Key Components:
- Promotion Gap Risk Scoring
- Retention Opportunity Identification
- Cluster-based Career Profiling
- Manager Stability Analysis
- Training Need Identification
- Department-level Insights

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import logging
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CareerAnalysis:
    """
    Comprehensive career analysis and risk scoring.
    
    This class analyzes career progression patterns, identifies promotion stagnation,
    and calculates retention risk scores to guide HR interventions.
    
    Attributes:
        df (pd.DataFrame): Employee data with engineered features
        cluster_labels (np.ndarray): Cluster assignments for each employee
        cluster_names (Dict): Mapping of cluster_id to cluster labels
        risk_scores (pd.DataFrame): Calculated risk and opportunity scores
        analysis_results (Dict): Comprehensive analysis results
    """
    
    def __init__(self, dataframe: pd.DataFrame, cluster_labels: Optional[np.ndarray] = None):
        """
        Initialize the CareerAnalysis class.
        
        Args:
            dataframe (pd.DataFrame): Employee data with engineered features
            cluster_labels (np.ndarray, optional): Cluster assignments for employees
        """
        self.df = dataframe.copy()
        self.cluster_labels = cluster_labels
        self.cluster_names = {}
        self.risk_scores = None
        self.analysis_results = {}
        
        # Add cluster column if provided
        if cluster_labels is not None:
            self.df['Cluster'] = cluster_labels
        
        logger.info(f"CareerAnalysis initialized with {len(self.df)} employees")
    
    def calculate_promotion_gap_score(self, 
                                     high_gap_threshold: float = 0.6,
                                     medium_gap_threshold: float = 0.3) -> pd.Series:
        """
        Calculate Promotion Gap Risk Score (0-1 scale).
        
        Based on:
        - YearsSinceLastPromotion
        - Ratio relative to tenure
        - Deviation from department average
        
        Args:
            high_gap_threshold (float): Threshold for HIGH risk (0-1)
            medium_gap_threshold (float): Threshold for MEDIUM risk (0-1)
        
        Returns:
            pd.Series: Promotion gap risk scores (0-1)
        """
        if 'PromotionGapRatio' not in self.df.columns:
            logger.warning("PromotionGapRatio column not found")
            return None
        
        promotion_gap = self.df['PromotionGapRatio'].copy()
        
        # Normalize to 0-1 scale
        scaler = MinMaxScaler()
        gap_score = scaler.fit_transform(promotion_gap.values.reshape(-1, 1)).flatten()
        
        # Calculate risk labels
        risk_labels = []
        for score in gap_score:
            if score >= high_gap_threshold:
                risk_labels.append('HIGH')
            elif score >= medium_gap_threshold:
                risk_labels.append('MEDIUM')
            else:
                risk_labels.append('LOW')
        
        self.analysis_results['promotion_gap_score'] = {
            'values': gap_score,
            'risk_labels': risk_labels,
            'high_threshold': high_gap_threshold,
            'medium_threshold': medium_gap_threshold,
            'high_risk_count': risk_labels.count('HIGH'),
            'medium_risk_count': risk_labels.count('MEDIUM'),
            'low_risk_count': risk_labels.count('LOW')
        }
        
        logger.info(f"Promotion Gap Score calculated: "
                   f"HIGH={risk_labels.count('HIGH')}, "
                   f"MEDIUM={risk_labels.count('MEDIUM')}, "
                   f"LOW={risk_labels.count('LOW')}")
        
        return pd.Series(gap_score, index=self.df.index)
    
    def calculate_stagnation_risk_score(self) -> pd.Series:
        """
        Calculate Role Stagnation Risk Score (0-1 scale).
        
        Combines:
        - RoleStagnationIndex
        - Years in current role
        - Lack of training/development
        
        Returns:
            pd.Series: Stagnation risk scores (0-1)
        """
        if 'RoleStagnationIndex' not in self.df.columns:
            logger.warning("RoleStagnationIndex column not found")
            return None
        
        stagnation_base = self.df['RoleStagnationIndex'].copy()
        
        # Factor in training (negative indicator)
        if 'TrainingIntensityScore' in self.df.columns:
            # Low training increases stagnation risk
            training_penalty = 1 - self.df['TrainingIntensityScore']
            stagnation_score = 0.7 * stagnation_base + 0.3 * training_penalty
        else:
            stagnation_score = stagnation_base
        
        # Normalize to 0-1
        scaler = MinMaxScaler()
        stagnation_score = scaler.fit_transform(stagnation_score.values.reshape(-1, 1)).flatten()
        
        self.analysis_results['stagnation_risk_score'] = {
            'values': stagnation_score,
            'high_risk_employees': (stagnation_score > 0.6).sum()
        }
        
        logger.info(f"Stagnation Risk Score calculated: "
                   f"{(stagnation_score > 0.6).sum()} high-risk employees")
        
        return pd.Series(stagnation_score, index=self.df.index)
    
    def calculate_retention_opportunity_index(self,
                                            promotion_gap_weight: float = 0.4,
                                            stagnation_weight: float = 0.3,
                                            attrition_weight: float = 0.3) -> pd.Series:
        """
        Calculate Retention Opportunity Index (0-1 scale).
        
        Identifies employees who:
        - Show career stagnation signals
        - Are not yet disengaged
        - Have not left (not high attrition risk)
        - Are viable intervention targets
        
        Args:
            promotion_gap_weight (float): Weight for promotion gap risk
            stagnation_weight (float): Weight for stagnation risk
            attrition_weight (float): Weight for current attrition status
        
        Returns:
            pd.Series: Retention opportunity scores (0-1)
        """
        # Get or calculate component scores
        if 'promotion_gap_score' not in self.analysis_results:
            promotion_gap = self.calculate_promotion_gap_score()
        else:
            promotion_gap = pd.Series(self.analysis_results['promotion_gap_score']['values'],
                                    index=self.df.index)
        
        if 'stagnation_risk_score' not in self.analysis_results:
            stagnation = self.calculate_stagnation_risk_score()
        else:
            stagnation = pd.Series(self.analysis_results['stagnation_risk_score']['values'],
                                 index=self.df.index)
        
        # Attrition component (employees still with company but at risk)
        if 'Attrition' in self.df.columns:
            # Inverse attrition: higher for employees who haven't left
            attrition_score = 1 - self.df['Attrition'].astype(float)
        else:
            # If no attrition info, assume all are retention opportunities
            attrition_score = pd.Series(np.ones(len(self.df)), index=self.df.index)
        
        # Combine weighted scores
        retention_opportunity = (promotion_gap_weight * promotion_gap +
                               stagnation_weight * stagnation +
                               attrition_weight * attrition_score)
        
        # Normalize to 0-1
        scaler = MinMaxScaler()
        retention_opportunity = scaler.fit_transform(
            retention_opportunity.values.reshape(-1, 1)
        ).flatten()
        
        # Calculate percentiles for intervention priority
        opportunity_labels = []
        for score in retention_opportunity:
            if score >= 0.7:
                opportunity_labels.append('CRITICAL')
            elif score >= 0.5:
                opportunity_labels.append('HIGH')
            elif score >= 0.3:
                opportunity_labels.append('MEDIUM')
            else:
                opportunity_labels.append('LOW')
        
        self.analysis_results['retention_opportunity_index'] = {
            'values': retention_opportunity,
            'opportunity_labels': opportunity_labels,
            'critical_count': opportunity_labels.count('CRITICAL'),
            'high_count': opportunity_labels.count('HIGH'),
            'medium_count': opportunity_labels.count('MEDIUM'),
            'low_count': opportunity_labels.count('LOW')
        }
        
        logger.info(f"Retention Opportunity Index calculated: "
                   f"CRITICAL={opportunity_labels.count('CRITICAL')}, "
                   f"HIGH={opportunity_labels.count('HIGH')}")
        
        return pd.Series(retention_opportunity, index=self.df.index)
    
    def identify_retention_target_employees(self,
                                           opportunity_threshold: float = 0.5,
                                           exclude_recent_promotions: bool = True) -> pd.DataFrame:
        """
        Identify employees who should be targeted for retention interventions.
        
        Criteria:
        - Retention Opportunity Index >= threshold
        - Haven't left the company
        - Show career stagnation signals
        - Recent promotion status considered
        
        Args:
            opportunity_threshold (float): Minimum opportunity score
            exclude_recent_promotions (bool): Exclude employees promoted in last year
        
        Returns:
            pd.DataFrame: Target employees with suggested interventions
        """
        if 'retention_opportunity_index' not in self.analysis_results:
            retention_scores = self.calculate_retention_opportunity_index()
        else:
            retention_scores = pd.Series(self.analysis_results['retention_opportunity_index']['values'],
                                       index=self.df.index)
        
        # Filter by threshold
        target_employees = self.df[retention_scores >= opportunity_threshold].copy()
        
        # Exclude if recently promoted
        if exclude_recent_promotions and 'YearsSinceLastPromotion' in target_employees.columns:
            target_employees = target_employees[
                target_employees['YearsSinceLastPromotion'] >= 1
            ]
        
        # Exclude if already left
        if 'Attrition' in target_employees.columns:
            target_employees = target_employees[target_employees['Attrition'] == 0]
        
        # Add scores
        target_employees['RetentionOpportunityScore'] = retention_scores[target_employees.index]
        
        # Rank by opportunity
        target_employees = target_employees.sort_values('RetentionOpportunityScore', ascending=False)
        
        logger.info(f"Identified {len(target_employees)} retention target employees")
        
        return target_employees
    
    def recommend_interventions(self, employee_id: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Recommend career interventions for retention.
        
        Personalized recommendations based on:
        - Promotion gap
        - Stagnation indicators
        - Training history
        - Manager tenure
        - Performance rating
        
        Args:
            employee_id (int, optional): Specific employee. If None, generates for all
        
        Returns:
            Dict: Recommended interventions by type
        """
        recommendations = {
            'Promotion': [],
            'Training': [],
            'Role Rotation': [],
            'Manager Meeting': [],
            'Career Planning': []
        }
        
        if employee_id is not None:
            employees_to_analyze = self.df[self.df.index == employee_id]
        else:
            # For target employees
            target_employees = self.identify_retention_target_employees()
            employees_to_analyze = target_employees
        
        for idx, employee in employees_to_analyze.iterrows():
            emp_recommendations = []
            
            # Promotion recommendation
            if 'YearsSinceLastPromotion' in employee and employee['YearsSinceLastPromotion'] >= 3:
                emp_recommendations.append(
                    f"Schedule promotion review (no promotion for {employee['YearsSinceLastPromotion']:.0f} years)"
                )
            
            # Training recommendation
            if 'TrainingTimesLastYear' in employee and employee['TrainingTimesLastYear'] < 2:
                emp_recommendations.append(
                    "Enroll in skill development/certification program"
                )
            
            # Role rotation recommendation
            if 'YearsInCurrentRole' in employee and employee['YearsInCurrentRole'] >= 4:
                emp_recommendations.append(
                    "Consider lateral role transition or cross-functional project assignment"
                )
            
            # Manager meeting recommendation
            if 'YearsWithCurrManager' in employee and employee['YearsWithCurrManager'] < 1:
                emp_recommendations.append(
                    "Schedule career development discussion with new manager"
                )
            elif 'YearsWithCurrManager' in employee and employee['YearsWithCurrManager'] >= 5:
                emp_recommendations.append(
                    "Review manager-employee relationship and consider mentoring opportunities"
                )
            
            # Career planning
            if 'PerformanceRating' in employee and employee['PerformanceRating'] >= 3:
                emp_recommendations.append(
                    "Develop personalized career development plan with high performer"
                )
        
        self.analysis_results['recommended_interventions'] = recommendations
        
        return recommendations
    
    def analyze_by_department(self) -> pd.DataFrame:
        """
        Analyze career stagnation and retention risk by department.
        
        Returns:
            pd.DataFrame: Department-level analysis
        """
        if 'Department' not in self.df.columns:
            logger.warning("Department column not found")
            return None
        
        if 'promotion_gap_score' not in self.analysis_results:
            self.calculate_promotion_gap_score()
        
        if 'retention_opportunity_index' not in self.analysis_results:
            self.calculate_retention_opportunity_index()
        
        promotion_gap = pd.Series(self.analysis_results['promotion_gap_score']['values'],
                                 index=self.df.index)
        retention_opp = pd.Series(self.analysis_results['retention_opportunity_index']['values'],
                                 index=self.df.index)
        
        dept_analysis = self.df.groupby('Department').agg({
            'EmployeeNumber' if 'EmployeeNumber' in self.df.columns else self.df.columns[0]: 'count'
        }).rename(columns={list(self.df.columns)[0]: 'TotalEmployees'})
        
        dept_analysis['AvgPromotionGapScore'] = self.df.groupby('Department').apply(
            lambda x: promotion_gap[x.index].mean()
        )
        
        dept_analysis['AvgRetentionOpportunity'] = self.df.groupby('Department').apply(
            lambda x: retention_opp[x.index].mean()
        )
        
        dept_analysis['HighRiskEmployees'] = self.df.groupby('Department').apply(
            lambda x: (promotion_gap[x.index] > 0.6).sum()
        )
        
        dept_analysis['HighRiskPercentage'] = (
            dept_analysis['HighRiskEmployees'] / dept_analysis['TotalEmployees'] * 100
        )
        
        # Sort by risk
        dept_analysis = dept_analysis.sort_values('HighRiskPercentage', ascending=False)
        
        self.analysis_results['department_analysis'] = dept_analysis
        
        logger.info(f"Department analysis completed: {len(dept_analysis)} departments")
        
        return dept_analysis
    
    def analyze_by_cluster(self, cluster_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """
        Analyze career characteristics and retention needs by career cluster.
        
        Args:
            cluster_names (Dict, optional): Mapping of cluster_id to cluster names
        
        Returns:
            pd.DataFrame: Cluster-level analysis
        """
        if 'Cluster' not in self.df.columns:
            logger.warning("Cluster column not found")
            return None
        
        if 'promotion_gap_score' not in self.analysis_results:
            self.calculate_promotion_gap_score()
        
        promotion_gap = pd.Series(self.analysis_results['promotion_gap_score']['values'],
                                 index=self.df.index)
        
        cluster_analysis = self.df.groupby('Cluster').agg({
            'EmployeeNumber' if 'EmployeeNumber' in self.df.columns else self.df.columns[0]: 'count'
        }).rename(columns={list(self.df.columns)[0]: 'Size'})
        
        cluster_analysis['AvgTenure'] = self.df.groupby('Cluster')['YearsAtCompany'].mean()
        cluster_analysis['AvgPromotionGap'] = self.df.groupby('Cluster')['YearsSinceLastPromotion'].mean()
        cluster_analysis['AvgTraining'] = self.df.groupby('Cluster')['TrainingTimesLastYear'].mean()
        cluster_analysis['HighPromotionGapRisk'] = self.df.groupby('Cluster').apply(
            lambda x: (promotion_gap[x.index] > 0.6).sum()
        )
        
        cluster_analysis['Percentage'] = cluster_analysis['Size'] / len(self.df) * 100
        
        if cluster_names:
            cluster_analysis['ClusterLabel'] = cluster_analysis.index.map(cluster_names)
        
        self.analysis_results['cluster_analysis'] = cluster_analysis
        
        logger.info(f"Cluster analysis completed: {len(cluster_analysis)} clusters")
        
        return cluster_analysis
    
    def analyze_manager_impact(self) -> pd.DataFrame:
        """
        Analyze impact of manager stability on career progression.
        
        Returns:
            pd.DataFrame: Manager impact analysis
        """
        if 'YearsWithCurrManager' not in self.df.columns:
            logger.warning("YearsWithCurrManager column not found")
            return None
        
        # Categorize by manager tenure
        def categorize_manager_tenure(years):
            if years < 1:
                return 'New (< 1 year)'
            elif years < 3:
                return 'Recent (1-3 years)'
            else:
                return 'Established (3+ years)'
        
        self.df['ManagerTenureCategory'] = self.df['YearsWithCurrManager'].apply(
            categorize_manager_tenure
        )
        
        manager_impact = self.df.groupby('ManagerTenureCategory').agg({
            'EmployeeNumber' if 'EmployeeNumber' in self.df.columns else self.df.columns[0]: 'count'
        }).rename(columns={list(self.df.columns)[0]: 'EmployeeCount'})
        
        manager_impact['AvgTenure'] = self.df.groupby('ManagerTenureCategory')['YearsAtCompany'].mean()
        manager_impact['AvgPromotionsSince'] = self.df.groupby('ManagerTenureCategory')[
            'YearsSinceLastPromotion'
        ].mean()
        manager_impact['AvgPerformance'] = self.df.groupby('ManagerTenureCategory')[
            'PerformanceRating'
        ].mean() if 'PerformanceRating' in self.df.columns else 0
        
        self.analysis_results['manager_impact'] = manager_impact
        
        logger.info(f"Manager impact analysis completed")
        
        return manager_impact
    
    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report.
        
        Returns:
            str: Formatted analysis report
        """
        report = []
        report.append("=" * 100)
        report.append("CAREER ANALYSIS REPORT - CareerGapAI")
        report.append("=" * 100)
        
        # Summary
        report.append(f"\nTOTAL EMPLOYEES ANALYZED: {len(self.df)}")
        
        # Promotion Gap Summary
        if 'promotion_gap_score' in self.analysis_results:
            summary = self.analysis_results['promotion_gap_score']
            report.append(f"\nPROMOTION GAP RISK SUMMARY:")
            report.append(f"  HIGH RISK: {summary['high_risk_count']} employees "
                        f"({summary['high_risk_count']/len(self.df)*100:.1f}%)")
            report.append(f"  MEDIUM RISK: {summary['medium_risk_count']} employees "
                        f"({summary['medium_risk_count']/len(self.df)*100:.1f}%)")
            report.append(f"  LOW RISK: {summary['low_risk_count']} employees "
                        f"({summary['low_risk_count']/len(self.df)*100:.1f}%)")
        
        # Retention Opportunity Summary
        if 'retention_opportunity_index' in self.analysis_results:
            summary = self.analysis_results['retention_opportunity_index']
            report.append(f"\nRETENTION OPPORTUNITY SUMMARY:")
            report.append(f"  CRITICAL: {summary['critical_count']} employees "
                        f"({summary['critical_count']/len(self.df)*100:.1f}%)")
            report.append(f"  HIGH: {summary['high_count']} employees "
                        f"({summary['high_count']/len(self.df)*100:.1f}%)")
        
        # Department Analysis
        if 'department_analysis' in self.analysis_results:
            report.append(f"\nDEPARTMENT-LEVEL INSIGHTS:")
            report.append(self.analysis_results['department_analysis'].to_string())
        
        # Cluster Analysis
        if 'cluster_analysis' in self.analysis_results:
            report.append(f"\nCLUSTER-LEVEL ANALYSIS:")
            report.append(self.analysis_results['cluster_analysis'].to_string())
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)


if __name__ == "__main__":
    """
    Example usage of the CareerAnalysis class.
    """
    # Load data with clusters
    # df = pd.read_csv('data/processed_data_with_clusters.csv')
    # cluster_labels = df['Cluster'].values
    
    # Initialize analysis
    # analyzer = CareerAnalysis(df, cluster_labels)
    
    # Calculate scores
    # analyzer.calculate_promotion_gap_score()
    # analyzer.calculate_stagnation_risk_score()
    # analyzer.calculate_retention_opportunity_index()
    
    # Identify targets
    # target_employees = analyzer.identify_retention_target_employees()
    
    # Department & Cluster analysis
    # dept_analysis = analyzer.analyze_by_department()
    # cluster_analysis = analyzer.analyze_by_cluster()
    
    # Generate report
    # report = analyzer.generate_analysis_report()
    
    print("CareerAnalysis module ready for use.")
