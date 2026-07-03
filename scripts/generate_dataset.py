"""
HR Analytics Dataset Generator
Generates 1000+ realistic employee records with career progression patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_hr_dataset(n_employees=1200):
    """Generate realistic HR analytics dataset"""
    np.random.seed(42)
    
    # Departments
    departments = ['Sales', 'Engineering', 'Product', 'Marketing', 'Finance', 'HR', 'Operations']
    
    # Positions
    positions = ['Junior', 'Senior', 'Lead', 'Manager', 'Director', 'VP', 'Executive']
    
    # Education levels
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'MBA']
    
    # Generate basic employee info
    employee_ids = [f'EMP{i:05d}' for i in range(1, n_employees + 1)]
    first_names = np.random.choice(['James', 'Sarah', 'Michael', 'Emily', 'David', 'Jennifer', 
                                     'Robert', 'Linda', 'William', 'Barbara', 'John', 'Mary',
                                     'Richard', 'Patricia', 'Thomas', 'Jennifer', 'Charles', 'Maria',
                                     'Christopher', 'Susan', 'Daniel', 'Jessica', 'Matthew', 'Sarah',
                                     'Mark', 'Karen', 'Donald', 'Nancy', 'Steven', 'Elizabeth'], n_employees)
    last_names = np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 
                                    'Martinez', 'Rodriguez', 'Lee', 'Miller', 'Davis', 'Wilson',
                                    'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson', 'Martin',
                                    'Patterson', 'Edwards', 'Collins', 'Reeves', 'Morris', 'Murphy',
                                    'Cook', 'Morgan', 'Peterson', 'Howard', 'Ward', 'Cox'], n_employees)
    
    # Career metrics with realistic distributions
    years_at_company = np.random.exponential(scale=5, size=n_employees) + 0.5
    years_in_current_role = np.random.exponential(scale=2.5, size=n_employees) + 0.5
    years_since_promotion = np.random.exponential(scale=2.5, size=n_employees)
    
    # Ensure promotion years don't exceed company tenure
    years_since_promotion = np.minimum(years_since_promotion, years_at_company - years_in_current_role)
    years_since_promotion = np.maximum(years_since_promotion, 0)
    
    # Performance ratings (normal distribution)
    performance_rating = np.random.normal(loc=3.5, scale=0.6, size=n_employees)
    performance_rating = np.clip(performance_rating, 2.0, 5.0)
    
    # Training hours (correlated with performance and company)
    base_training = 20 + performance_rating * 5 + np.random.normal(0, 5, n_employees)
    training_hours_per_year = np.maximum(base_training, 5)
    
    # Salary (correlated with years at company, performance, and position)
    base_salary = 50000
    salary = base_salary + (years_at_company * 3000) + (performance_rating * 8000) + \
             np.random.normal(0, 5000, n_employees)
    salary = np.maximum(salary, 40000)
    
    # Age (correlated with years at company)
    age = 25 + years_at_company * 1.5 + np.random.normal(0, 3, n_employees)
    age = np.clip(age, 20, 70)
    
    # Manager tenure (related to their seniority)
    years_with_current_manager = np.minimum(years_at_company, np.random.exponential(scale=3, size=n_employees) + 0.5)
    
    # Gender distribution
    gender = np.random.choice(['Male', 'Female'], size=n_employees, p=[0.55, 0.45])
    
    # Department assignment
    department = np.random.choice(departments, size=n_employees)
    
    # Position level (correlated with tenure and performance)
    position_idx = np.minimum(
        np.floor((years_at_company / 15) * len(positions) + (performance_rating - 2) * 0.5).astype(int),
        len(positions) - 1
    )
    position = np.array(positions)[position_idx]
    
    # Education level (correlated with position)
    education_weights = np.array([0.1, 0.4, 0.35, 0.1, 0.05])
    if 'Manager' in position or 'Director' in position or 'VP' in position:
        education_weights = np.array([0.05, 0.25, 0.4, 0.15, 0.15])
    education = np.random.choice(education_levels, size=n_employees, p=education_weights)
    
    # Job satisfaction (inverse correlation with promotion gap, correlated with pay)
    promotion_gap = years_since_promotion / (years_at_company + 1)
    job_satisfaction = 4.0 - (promotion_gap * 1.5) + (salary / 100000 * 0.5) + np.random.normal(0, 0.3, n_employees)
    job_satisfaction = np.clip(job_satisfaction, 1.0, 5.0)
    
    # Attrition risk (higher with promotion gap, lower with satisfaction)
    attrition_risk = (promotion_gap * 0.4) - (job_satisfaction * 0.1) + np.random.normal(0, 0.15, n_employees)
    attrition_risk = np.clip(attrition_risk, 0.0, 1.0)
    
    # Remote work eligibility
    remote_work = np.random.choice([0, 1], size=n_employees, p=[0.3, 0.7])
    
    # Bonus percentage (correlated with performance and position)
    bonus_percentage = (performance_rating / 5.0) * 0.3 + np.random.normal(0, 0.05, n_employees)
    bonus_percentage = np.clip(bonus_percentage * 100, 0, 50)
    
    # Create DataFrame
    df = pd.DataFrame({
        'EmployeeID': employee_ids,
        'FirstName': first_names,
        'LastName': last_names,
        'Department': department,
        'Position': position,
        'Gender': gender,
        'Age': np.round(age, 1),
        'EducationLevel': education,
        'YearsAtCompany': np.round(years_at_company, 2),
        'YearsInCurrentRole': np.round(years_in_current_role, 2),
        'YearsSinceLastPromotion': np.round(years_since_promotion, 2),
        'YearsWithCurrentManager': np.round(years_with_current_manager, 2),
        'AnnualSalary': np.round(salary, 0).astype(int),
        'BonusPercentage': np.round(bonus_percentage, 1),
        'PerformanceRating': np.round(performance_rating, 2),
        'JobSatisfaction': np.round(job_satisfaction, 2),
        'TrainingHoursPerYear': np.round(training_hours_per_year, 1),
        'AttritionRisk': np.round(attrition_risk, 3),
        'RemoteWork': remote_work,
    })
    
    return df

# Generate and save dataset
if __name__ == "__main__":
    df = generate_hr_dataset(n_employees=1200)
    
    # Save dataset
    output_path = '../data/hr_analytics.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} employee records")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Dataset Summary:")
    print(f"  Departments: {df['Department'].nunique()}")
    print(f"  Avg Salary: ${df['AnnualSalary'].mean():,.0f}")
    print(f"  Avg Tenure: {df['YearsAtCompany'].mean():.1f} years")
    print(f"  Avg Performance: {df['PerformanceRating'].mean():.2f}/5.0")
    print(f"\n🎯 Columns: {len(df.columns)}")
    print(df.head(10))
