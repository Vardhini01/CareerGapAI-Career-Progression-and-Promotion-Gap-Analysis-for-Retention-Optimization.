"""
AI Career Advisor Module - CareerGapAI

Generates personalized career recommendations for employees based on their 
profile, performance, and risk scores.

Recommendations include:
- Promotion Review Recommended
- Leadership Training Recommended
- Internal Role Rotation Suggested
- Manager Discussion Recommended
- Career Development Plan Needed

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CareerAdvisor:
    """AI-powered career recommendations engine."""
    
    RECOMMENDATION_TEMPLATES = {
        'promotion': {
            'text': '🚨 Promotion Review Recommended',
            'description': 'Employee is overdue for promotion consideration',
            'priority': 'High'
        },
        'training': {
            'text': '🎓 Leadership Training Recommended',
            'description': 'Investment in skill development needed',
            'priority': 'High'
        },
        'rotation': {
            'text': '🔄 Internal Role Rotation Suggested',
            'description': 'Fresh challenge within organization',
            'priority': 'Medium'
        },
        'manager_discussion': {
            'text': '💬 Manager Discussion Recommended',
            'description': 'Explore career aspirations and support',
            'priority': 'High'
        },
        'development_plan': {
            'text': '📈 Career Development Plan Needed',
            'description': 'Structured pathway for career growth',
            'priority': 'Medium'
        },
        'engagement': {
            'text': '💪 Employee Engagement Initiative',
            'description': 'Boost motivation and satisfaction',
            'priority': 'High'
        },
        'mentorship': {
            'text': '👥 Mentorship Program',
            'description': 'Peer mentoring or executive sponsorship',
            'priority': 'Medium'
        },
        'flexibility': {
            'text': '⏰ Work-Life Balance Review',
            'description': 'Explore flexible work arrangements',
            'priority': 'High'
        }
    }
    
    @staticmethod
    def generate_recommendations(employee_row: pd.Series, risk_score: float) -> List[Dict]:
        """
        Generate personalized recommendations for an employee.
        
        Args:
            employee_row: Pandas Series with employee data
            risk_score: Retention risk score (0-100)
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Extract key metrics
            years_since_promo = float(employee_row.get('YearsSinceLastPromotion', 0))
            years_in_role = float(employee_row.get('YearsInCurrentRole', 0))
            training_times = float(employee_row.get('TrainingTimesLastYear', 0))
            job_satisfaction = float(employee_row.get('JobSatisfaction', 3))
            work_life_balance = float(employee_row.get('WorkLifeBalance', 3))
            years_with_manager = float(employee_row.get('YearsWithCurrManager', 0))
            performance_rating = float(employee_row.get('PerformanceRating', 3))
            
            # HIGH RISK (65+) - Urgent interventions needed
            if risk_score >= 65:
                if years_since_promo > 5:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['promotion'].copy()
                    rec['reason'] = f"No promotion in {years_since_promo:.0f} years"
                    recommendations.append(rec)
                
                if job_satisfaction <= 2:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['manager_discussion'].copy()
                    rec['reason'] = "Low job satisfaction detected"
                    recommendations.append(rec)
                
                if work_life_balance <= 2:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['flexibility'].copy()
                    rec['reason'] = "Work-life balance concerns"
                    recommendations.append(rec)
                
                if years_in_role > 4 and training_times < 2:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['training'].copy()
                    rec['reason'] = f"Stagnant for {years_in_role:.0f} years with minimal training"
                    recommendations.append(rec)
            
            # MEDIUM RISK (35-65) - Development focus
            elif risk_score >= 35:
                if years_since_promo > 3:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['development_plan'].copy()
                    rec['reason'] = "Career pathway planning needed"
                    recommendations.append(rec)
                
                if training_times < 2:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['training'].copy()
                    rec['reason'] = "Skill enhancement opportunity"
                    recommendations.append(rec)
                
                if job_satisfaction <= 3:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['engagement'].copy()
                    rec['reason'] = "Moderate satisfaction - engagement opportunity"
                    recommendations.append(rec)
                
                if performance_rating >= 4 and years_in_role > 2:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['mentorship'].copy()
                    rec['reason'] = "High performer - consider mentor role"
                    recommendations.append(rec)
            
            # LOW RISK (0-35) - Growth opportunities
            else:
                if performance_rating >= 4:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['mentorship'].copy()
                    rec['reason'] = "Strong performer - expand leadership impact"
                    recommendations.append(rec)
                
                if training_times >= 3:
                    rec = CareerAdvisor.RECOMMENDATION_TEMPLATES['development_plan'].copy()
                    rec['reason'] = "Committed to learning - formalize development"
                    recommendations.append(rec)
            
            # Limit to top 4 recommendations
            return recommendations[:4] if recommendations else [
                {
                    'text': '✅ Continue Current Path',
                    'reason': 'No immediate action needed',
                    'priority': 'Low'
                }
            ]
        
        except Exception as e:
            return [
                {
                    'text': '📋 Review Career Status',
                    'reason': 'Schedule career discussion',
                    'priority': 'Medium'
                }
            ]
    
    @staticmethod
    def get_employee_archetype(employee_row: pd.Series, risk_score: float) -> str:
        """
        Classify employee into archetype.
        
        Args:
            employee_row: Employee data
            risk_score: Retention risk score
            
        Returns:
            Employee archetype label
        """
        performance_rating = float(employee_row.get('PerformanceRating', 3))
        job_satisfaction = float(employee_row.get('JobSatisfaction', 3))
        years_since_promo = float(employee_row.get('YearsSinceLastPromotion', 0))
        
        if performance_rating >= 4 and risk_score < 35:
            return "🌟 High Performer"
        elif risk_score >= 65 and job_satisfaction <= 2:
            return "⚠️ At-Risk Talent"
        elif years_since_promo > 4 and risk_score > 45:
            return "🔄 Career Stalled"
        elif job_satisfaction <= 2:
            return "💭 Disengaged"
        else:
            return "📊 Steady Performer"
    
    @staticmethod
    def get_priority_level(risk_score: float) -> str:
        """
        Get action priority level.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Priority level string
        """
        if risk_score >= 75:
            return "🚨 URGENT - Within 1 week"
        elif risk_score >= 65:
            return "⚠️ HIGH - Within 2 weeks"
        elif risk_score >= 50:
            return "📅 MEDIUM - Within 1 month"
        else:
            return "💚 LOW - Regular monitoring"
    
    @staticmethod
    def get_next_steps(employee_row: pd.Series, risk_score: float, 
                      recommendations: List[Dict]) -> List[str]:
        """
        Get concrete next steps for the employee.
        
        Args:
            employee_row: Employee data
            risk_score: Risk score
            recommendations: List of recommendations
            
        Returns:
            List of actionable next steps
        """
        next_steps = []
        
        try:
            if risk_score >= 65:
                next_steps.append("1. Schedule 1-on-1 with manager within 3 days")
                next_steps.append("2. Explore open opportunities within organization")
                next_steps.append("3. Discuss career aspirations and retention concerns")
                next_steps.append("4. Develop 30-60-90 day action plan")
            
            elif risk_score >= 35:
                next_steps.append("1. Enroll in relevant training/development program")
                next_steps.append("2. Schedule monthly career development check-ins")
                next_steps.append("3. Identify growth opportunities within department")
                next_steps.append("4. Build personalized development roadmap")
            
            else:
                next_steps.append("1. Continue current performance trajectory")
                next_steps.append("2. Explore leadership opportunities or mentoring roles")
                next_steps.append("3. Plan advanced skill development")
                next_steps.append("4. Maintain engagement through recognition programs")
            
            return next_steps
        
        except:
            return ["1. Schedule career development discussion with manager"]
    
    @staticmethod
    def generate_all_recommendations(df: pd.DataFrame, risk_scores: pd.Series) -> pd.DataFrame:
        """
        Generate recommendations for all employees in dataframe.
        
        Args:
            df: Employee dataframe
            risk_scores: Series with risk scores for each employee
            
        Returns:
            Dataframe with recommendations added
        """
        df = df.copy()
        
        recommendations_list = []
        archetypes = []
        priorities = []
        
        for idx, row in df.iterrows():
            risk_score = risk_scores.iloc[idx]
            recs = CareerAdvisor.generate_recommendations(row, risk_score)
            archetype = CareerAdvisor.get_employee_archetype(row, risk_score)
            priority = CareerAdvisor.get_priority_level(risk_score)
            
            # Format recommendations as string
            rec_text = "; ".join([r['text'] for r in recs])
            recommendations_list.append(rec_text)
            archetypes.append(archetype)
            priorities.append(priority)
        
        df['Recommendations'] = recommendations_list
        df['Archetype'] = archetypes
        df['ActionPriority'] = priorities
        
        return df
