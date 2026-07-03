"""
Palo Alto Networks HR Analytics Dataset Generator
Generates 1000+ realistic employee records with career progression, promotion gaps, and attrition patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_pa_dataset(n_employees=1200, random_seed=42):
    """Generate realistic Palo Alto Networks HR dataset"""
    np.random.seed(random_seed)
    
    # Departments
    departments = ['R&D', 'Sales', 'HR', 'Finance', 'Operations']
    job_roles = {
        'R&D': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Research Lead', 'Technical Architect'],
        'Sales': ['Sales Executive', 'Account Manager', 'Sales Manager', 'Regional Director', 'VP Sales'],
        'HR': ['HR Specialist', 'Recruiter', 'HR Manager', 'HR Director', 'Chief People Officer'],
        'Finance': ['Analyst', 'Senior Analyst', 'Financial Manager', 'Controller', 'CFO'],
        'Operations': ['Operations Specialist', 'Process Engineer', 'Operations Manager', 'Director of Ops', 'VP Operations']
    }
    
    business_travel = ['Non-Travel', 'Travel Rarely', 'Travel Frequently']
    education_levels = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
    education_fields = ['Life Sciences', 'Medical', 'Technical Degree', 'Business', 'Other']
    marital_status = ['Single', 'Married', 'Divorced']
    genders = ['Male', 'Female']
    
    data = {
        'Age': [],
        'Attrition': [],
        'BusinessTravel': [],
        'DailyRate': [],
        'Department': [],
        'DistanceFromHome': [],
        'Education': [],
        'EducationField': [],
        'EnvironmentSatisfaction': [],
        'Gender': [],
        'HourlyRate': [],
        'JobInvolvement': [],
        'JobLevel': [],
        'JobRole': [],
        'JobSatisfaction': [],
        'MaritalStatus': [],
        'MonthlyIncome': [],
        'MonthlyRate': [],
        'NumCompaniesWorked': [],
        'OverTime': [],
        'PercentSalaryHike': [],
        'PerformanceRating': [],
        'RelationshipSatisfaction': [],
        'StockOptionLevel': [],
        'TotalWorkingYears': [],
        'TrainingTimesLastYear': [],
        'WorkLifeBalance': [],
        'YearsAtCompany': [],
        'YearsInCurrentRole': [],
        'YearsSinceLastPromotion': [],
        'YearsWithCurrManager': [],
        'PromotionGapRatio': [],
        'RoleStagnationIndex': [],
        'TrainingIntensityScore': [],
        'ManagerStabilityIndicator': []
    }
    
    for i in range(n_employees):
        # Basic demographics
        age = np.random.normal(loc=38, scale=12)
        age = np.clip(age, 22, 65)
        
        gender = np.random.choice(genders)
        marital = np.random.choice(marital_status)
        distance = np.random.exponential(scale=10) + 1
        distance = np.clip(distance, 1, 30)
        
        # Education
        education = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.35, 0.20, 0.05])
        educ_field = np.random.choice(education_fields)
        
        # Career details
        dept = np.random.choice(departments)
        total_years = np.random.normal(loc=12, scale=8)
        total_years = np.clip(total_years, 0, 50)
        
        years_at_company = np.random.exponential(scale=6)
        years_at_company = np.clip(years_at_company, 0.5, min(total_years, 30))
        
        job_level = np.random.choice([1, 2, 3, 4, 5], p=[0.25, 0.30, 0.25, 0.15, 0.05])
        job_role = np.random.choice(job_roles[dept])
        
        years_in_role = np.random.exponential(scale=3)
        years_in_role = np.clip(years_in_role, 0.5, years_at_company)
        
        # Promotion gap (key metric)
        # Create correlation: longer in role -> more likely to have promotion gap
        base_gap = np.random.exponential(scale=2)
        gap_multiplier = (years_in_role / (years_at_company + 1))
        years_since_promo = np.clip(base_gap * gap_multiplier, 0, years_at_company - 0.5)
        
        years_with_manager = np.random.exponential(scale=3)
        years_with_manager = np.clip(years_with_manager, 0.5, years_at_company)
        
        # Work metrics
        job_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.35, 0.25])
        env_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.12, 0.28, 0.40, 0.20])
        relation_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.10, 0.30, 0.35, 0.25])
        job_involvement = np.random.choice([1, 2, 3, 4], p=[0.20, 0.25, 0.35, 0.20])
        work_life_balance = np.random.choice([1, 2, 3, 4], p=[0.25, 0.30, 0.30, 0.15])
        
        # Performance and development
        performance = np.random.choice([1, 2, 3, 4], p=[0.05, 0.10, 0.55, 0.30])
        training_times = np.random.poisson(lam=2.5)
        training_times = max(0, training_times)
        
        salary_hike = np.random.normal(loc=2.5, scale=1.2)
        salary_hike = np.clip(salary_hike, 0, 8)
        
        stock_option = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
        
        # Compensation
        hourly_rate = np.random.exponential(scale=50) + 20
        hourly_rate = np.clip(hourly_rate, 20, 150)
        
        daily_rate = hourly_rate * 8 + np.random.normal(0, 50)
        daily_rate = np.clip(daily_rate, 100, 1500)
        
        monthly_rate = daily_rate * 21 + np.random.normal(0, 300)
        monthly_rate = np.clip(monthly_rate, 1000, 10000)
        
        # Base salary depends on job level, department, and experience
        base_salary = {1: 50000, 2: 70000, 3: 95000, 4: 130000, 5: 180000}
        monthly_income = base_salary[job_level] / 12
        monthly_income += (total_years * 1000)
        monthly_income += (performance - 2) * 5000
        monthly_income = np.clip(monthly_income, 30000/12, 300000/12)
        
        # Work patterns
        business_travel_freq = np.random.choice(business_travel, p=[0.40, 0.35, 0.25])
        overtime = np.random.choice(['No', 'Yes'], p=[0.70, 0.30])
        num_companies = np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.30, 0.15, 0.08, 0.02])
        
        # ATTRITION LOGIC: More likely if high promotion gap + low satisfaction
        promotion_gap_risk = (years_since_promo / (years_at_company + 1))
        satisfaction_avg = (job_satisfaction + env_satisfaction + relation_satisfaction) / 3
        
        attrition_prob = 0.05  # Base probability
        attrition_prob += (promotion_gap_risk * 0.25)  # Promotion gap increases risk
        attrition_prob -= (satisfaction_avg / 4 * 0.15)  # Satisfaction decreases risk
        attrition_prob += (0.1 if work_life_balance <= 2 else 0)  # Work-life balance
        attrition_prob = np.clip(attrition_prob, 0.01, 0.50)
        
        attrition = 1 if np.random.random() < attrition_prob else 0
        
        # Feature Engineering metrics
        promotion_gap_ratio = years_since_promo / (years_at_company + 1)
        role_stagnation = years_in_role / (years_at_company + 1)
        training_intensity = training_times / (total_years + 1)
        manager_stability = years_with_manager / (years_at_company + 1)
        
        # Append data
        data['Age'].append(int(age))
        data['Attrition'].append(attrition)
        data['BusinessTravel'].append(business_travel_freq)
        data['DailyRate'].append(int(daily_rate))
        data['Department'].append(dept)
        data['DistanceFromHome'].append(int(distance))
        data['Education'].append(education)
        data['EducationField'].append(educ_field)
        data['EnvironmentSatisfaction'].append(int(env_satisfaction))
        data['Gender'].append(gender)
        data['HourlyRate'].append(int(hourly_rate))
        data['JobInvolvement'].append(int(job_involvement))
        data['JobLevel'].append(int(job_level))
        data['JobRole'].append(job_role)
        data['JobSatisfaction'].append(int(job_satisfaction))
        data['MaritalStatus'].append(marital)
        data['MonthlyIncome'].append(int(monthly_income * 1000))
        data['MonthlyRate'].append(int(monthly_rate))
        data['NumCompaniesWorked'].append(int(num_companies))
        data['OverTime'].append(overtime)
        data['PercentSalaryHike'].append(round(salary_hike, 2))
        data['PerformanceRating'].append(int(performance))
        data['RelationshipSatisfaction'].append(int(relation_satisfaction))
        data['StockOptionLevel'].append(int(stock_option))
        data['TotalWorkingYears'].append(int(total_years))
        data['TrainingTimesLastYear'].append(int(training_times))
        data['WorkLifeBalance'].append(int(work_life_balance))
        data['YearsAtCompany'].append(round(years_at_company, 1))
        data['YearsInCurrentRole'].append(round(years_in_role, 1))
        data['YearsSinceLastPromotion'].append(round(years_since_promo, 1))
        data['YearsWithCurrManager'].append(round(years_with_manager, 1))
        data['PromotionGapRatio'].append(round(promotion_gap_ratio, 3))
        data['RoleStagnationIndex'].append(round(role_stagnation, 3))
        data['TrainingIntensityScore'].append(round(training_intensity, 3))
        data['ManagerStabilityIndicator'].append(round(manager_stability, 3))
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    print("Generating Palo Alto Networks HR Analytics Dataset...")
    print("=" * 70)
    
    # Generate dataset
    df = generate_pa_dataset(n_employees=1200)
    
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    output_path = 'data/hr_analytics.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Records: {len(df)}")
    print(f"📁 Location: {output_path}")
    print(f"\nDataset Summary:")
    print(f"  • Total Employees: {len(df)}")
    print(f"  • Attrition Rate: {df['Attrition'].sum() / len(df) * 100:.1f}%")
    print(f"  • Departments: {df['Department'].nunique()}")
    print(f"  • Job Roles: {df['JobRole'].nunique()}")
    print(f"  • Avg Tenure: {df['YearsAtCompany'].mean():.1f} years")
    print(f"  • Avg Promotion Gap: {df['YearsSinceLastPromotion'].mean():.1f} years")
    print(f"  • High Risk Employees: {(df['PromotionGapRatio'] > 0.5).sum()}")
    
    print(f"\n{'='*70}")
    print("Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
