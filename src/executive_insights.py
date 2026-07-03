"""
Executive Insights Module - CareerGapAI

Generates high-level business insights from HR analytics data.
Provides actionable intelligence for executive decision-making.

Insights generated:
- Department with highest promotion gap
- Highest risk employee segment
- Impact of training on retention
- Manager stability observations
- Compensation fairness analysis
- Career progression patterns

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ExecutiveInsights:
    """Generate strategic business insights from HR data."""
    
    @staticmethod
    def get_promotion_gap_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze promotion gaps across organization.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with promotion gap insights
        """
        insights = []
        
        # Overall promotion gap
        avg_years_since_promo = df['YearsSinceLastPromotion'].mean()
        
        if avg_years_since_promo > 3:
            insights.append({
                'title': '📊 High Promotion Gap Alert',
                'value': f'{avg_years_since_promo:.1f} years average',
                'description': f'Employees wait {avg_years_since_promo:.1f} years on average for promotion',
                'recommendation': 'Review promotion criteria and create clear advancement pathways',
                'severity': 'High' if avg_years_since_promo > 4 else 'Medium'
            })
        
        # Department with largest gap
        dept_promo_gap = df.groupby('Department')['YearsSinceLastPromotion'].agg(['mean', 'count']).reset_index()
        worst_dept = dept_promo_gap.loc[dept_promo_gap['mean'].idxmax()]
        
        if worst_dept['mean'] > avg_years_since_promo * 1.3:
            insights.append({
                'title': f"🚨 {worst_dept['Department']} - Promotion Bottleneck",
                'value': f"{worst_dept['mean']:.1f} years ({int(worst_dept['count'])} employees)",
                'description': f"{worst_dept['Department']} has significant promotion delays",
                'recommendation': f"Prioritize career development for {worst_dept['Department']}'s top talent",
                'severity': 'High'
            })
        
        return {'insights': insights, 'avg_gap': avg_years_since_promo}
    
    @staticmethod
    def get_risk_segment_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze at-risk employee segments.
        
        Args:
            df: Employee dataframe with RiskScore column
            
        Returns:
            Dictionary with risk segment insights
        """
        insights = []
        
        if 'RiskScore' not in df.columns:
            return {'insights': insights, 'high_risk_pct': 0}
        
        high_risk = df[df['RiskScore'] >= 65]
        high_risk_pct = len(high_risk) / len(df) * 100
        
        # Overall risk alert
        if high_risk_pct > 15:
            insights.append({
                'title': '⚠️ High Attrition Risk',
                'value': f'{high_risk_pct:.1f}% of workforce',
                'description': f'{len(high_risk)} employees at high risk of departure',
                'recommendation': 'Implement retention initiatives immediately',
                'severity': 'Critical' if high_risk_pct > 25 else 'High'
            })
        
        # Risk by department
        dept_high_risk = high_risk.groupby('Department').size().reset_index(name='count')
        dept_total = df.groupby('Department').size().reset_index(name='total')
        dept_risk_pct = dept_high_risk.merge(dept_total, on='Department')
        dept_risk_pct['pct'] = dept_risk_pct['count'] / dept_risk_pct['total'] * 100
        
        if len(dept_risk_pct) > 0:
            worst_dept = dept_risk_pct.loc[dept_risk_pct['pct'].idxmax()]
            if worst_dept['pct'] > 20:
                insights.append({
                    'title': f"🔴 {worst_dept['Department']} - Critical Risk",
                    'value': f"{worst_dept['pct']:.0f}% of department ({int(worst_dept['count'])} people)",
                    'description': f"{worst_dept['Department']} has highest concentration of at-risk employees",
                    'recommendation': f"Conduct engagement survey in {worst_dept['Department']} immediately",
                    'severity': 'Critical'
                })
        
        return {'insights': insights, 'high_risk_pct': high_risk_pct}
    
    @staticmethod
    def get_training_impact_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze impact of training on retention/satisfaction.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with training impact insights
        """
        insights = []
        
        # Employees with training vs without
        trained_employees = df[df['TrainingTimesLastYear'] >= 2]
        untrained_employees = df[df['TrainingTimesLastYear'] < 2]
        
        avg_satisfaction_trained = trained_employees['JobSatisfaction'].mean()
        avg_satisfaction_untrained = untrained_employees['JobSatisfaction'].mean()
        
        satisfaction_diff = avg_satisfaction_trained - avg_satisfaction_untrained
        
        if satisfaction_diff > 0.3:
            insights.append({
                'title': '📚 Training Boosts Satisfaction',
                'value': f'+{satisfaction_diff:.2f} points',
                'description': f'Employees with 2+ trainings/year show {satisfaction_diff:.2f} higher satisfaction',
                'recommendation': 'Increase training budget - ROI evident in satisfaction metrics',
                'severity': 'Positive'
            })
        
        # Training coverage
        trained_pct = len(trained_employees) / len(df) * 100
        
        if trained_pct < 40:
            insights.append({
                'title': '📉 Low Training Participation',
                'value': f'{trained_pct:.0f}% of workforce',
                'description': f'Only {trained_pct:.0f}% receiving 2+ trainings annually',
                'recommendation': 'Expand training programs and encourage participation',
                'severity': 'Medium'
            })
        
        return {'insights': insights, 'trained_pct': trained_pct, 'satisfaction_diff': satisfaction_diff}
    
    @staticmethod
    def get_manager_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze manager stability and team dynamics.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with manager insights
        """
        insights = []
        
        # Manager turnover risk
        avg_years_with_manager = df['YearsWithCurrManager'].mean()
        
        managers_high_turnover = df[df['YearsWithCurrManager'] < 1]
        turnover_risk_pct = len(managers_high_turnover) / len(df) * 100
        
        if turnover_risk_pct > 20:
            insights.append({
                'title': '⚠️ Manager Stability Concerns',
                'value': f'{turnover_risk_pct:.0f}% with new managers',
                'description': f'{len(managers_high_turnover)} employees have managers for <1 year',
                'recommendation': 'Prioritize manager retention and executive coaching',
                'severity': 'High'
            })
        
        # Average tenure with manager
        if avg_years_with_manager < 2:
            insights.append({
                'title': '📊 High Manager Turnover',
                'value': f'{avg_years_with_manager:.1f} years average',
                'description': 'Frequent manager changes disrupt team dynamics',
                'recommendation': 'Review manager retention programs and career paths',
                'severity': 'Medium'
            })
        
        # Manager span of control (if available)
        if 'ManagerID' in df.columns:
            manager_span = df.groupby('ManagerID').size()
            avg_span = manager_span.mean()
            max_span = manager_span.max()
            
            if max_span > 15:
                insights.append({
                    'title': '👥 Unbalanced Manager Load',
                    'value': f'{max_span} direct reports (avg: {avg_span:.0f})',
                    'description': 'Some managers oversee too many direct reports',
                    'recommendation': 'Rebalance organizational structure for better support',
                    'severity': 'Medium'
                })
        
        return {'insights': insights, 'avg_manager_tenure': avg_years_with_manager}
    
    @staticmethod
    def get_role_stagnation_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze employees stuck in same role.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with role stagnation insights
        """
        insights = []
        
        # Employees in same role 4+ years
        stagnant = df[df['YearsInCurrentRole'] >= 4]
        stagnant_pct = len(stagnant) / len(df) * 100
        
        if stagnant_pct > 20:
            insights.append({
                'title': '🔄 Role Stagnation Alert',
                'value': f'{stagnant_pct:.0f}% of workforce',
                'description': f'{len(stagnant)} employees in same role for 4+ years',
                'recommendation': 'Create rotation programs and lateral career paths',
                'severity': 'High'
            })
        
        # High performers stuck
        stagnant_high_performers = stagnant[stagnant['PerformanceRating'] >= 4]
        
        if len(stagnant_high_performers) > 0:
            insights.append({
                'title': '⭐ Retention Risk: High Performers Stalled',
                'value': f'{len(stagnant_high_performers)} top performers',
                'description': 'High performers in same role risk departure',
                'recommendation': 'Fast-track promotions or strategic opportunities',
                'severity': 'Critical'
            })
        
        return {'insights': insights, 'stagnant_pct': stagnant_pct}
    
    @staticmethod
    def get_satisfaction_insights(df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze job satisfaction patterns.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with satisfaction insights
        """
        insights = []
        
        low_satisfaction = df[df['JobSatisfaction'] <= 2]
        low_sat_pct = len(low_satisfaction) / len(df) * 100
        
        if low_sat_pct > 15:
            insights.append({
                'title': '😔 Low Satisfaction Alert',
                'value': f'{low_sat_pct:.0f}% of workforce',
                'description': f'{len(low_satisfaction)} employees report low job satisfaction',
                'recommendation': 'Conduct engagement survey and address root causes',
                'severity': 'High'
            })
        
        # Work-life balance issues
        poor_balance = df[df['WorkLifeBalance'] <= 2]
        balance_pct = len(poor_balance) / len(df) * 100
        
        if balance_pct > 20:
            insights.append({
                'title': '⏰ Work-Life Balance Crisis',
                'value': f'{balance_pct:.0f}% report poor balance',
                'description': 'Significant portion of workforce struggling with work-life balance',
                'recommendation': 'Implement flexible work policies and workload review',
                'severity': 'High'
            })
        
        return {'insights': insights, 'low_satisfaction_pct': low_sat_pct}
    
    @staticmethod
    def generate_all_insights(df: pd.DataFrame) -> List[Dict]:
        """
        Generate all executive insights.
        
        Args:
            df: Employee dataframe
            
        Returns:
            List of all insights sorted by severity
        """
        all_insights = []
        
        # Gather insights from all modules
        all_insights.extend(ExecutiveInsights.get_promotion_gap_insights(df)['insights'])
        all_insights.extend(ExecutiveInsights.get_risk_segment_insights(df)['insights'])
        all_insights.extend(ExecutiveInsights.get_training_impact_insights(df)['insights'])
        all_insights.extend(ExecutiveInsights.get_manager_insights(df)['insights'])
        all_insights.extend(ExecutiveInsights.get_role_stagnation_insights(df)['insights'])
        all_insights.extend(ExecutiveInsights.get_satisfaction_insights(df)['insights'])
        
        # Sort by severity
        severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Positive': 3, 'Low': 4}
        all_insights.sort(key=lambda x: severity_order.get(x.get('severity', 'Medium'), 5))
        
        # Limit to top 10
        return all_insights[:10]
    
    @staticmethod
    def get_quick_stats(df: pd.DataFrame) -> Dict[str, any]:
        """
        Get quick statistics for dashboard KPIs.
        
        Args:
            df: Employee dataframe
            
        Returns:
            Dictionary with key statistics
        """
        stats = {
            'total_employees': len(df),
            'avg_tenure': df['YearsAtCompany'].mean(),
            'avg_satisfaction': df['JobSatisfaction'].mean(),
            'turnover_rate': (df['Attrition'].sum() / len(df) * 100) if 'Attrition' in df.columns else 0,
            'high_performers': len(df[df['PerformanceRating'] >= 4]),
            'departments': df['Department'].nunique(),
            'avg_monthly_income': df['MonthlyIncome'].mean() if 'MonthlyIncome' in df.columns else 0
        }
        
        if 'RiskScore' in df.columns:
            stats['high_risk_employees'] = len(df[df['RiskScore'] >= 65])
            stats['avg_risk_score'] = df['RiskScore'].mean()
        
        return stats
