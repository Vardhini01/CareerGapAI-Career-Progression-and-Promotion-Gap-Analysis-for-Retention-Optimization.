"""
Generate Clean HR Employee Dataset
Simple, realistic employee data for testing
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 500 realistic employees
n = 500

data = {
    'EmployeeID': [f'EMP{i:04d}' for i in range(1, n+1)],
    'Name': np.random.choice([
        'James Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis', 'David Wilson',
        'Jennifer Martinez', 'Robert Garcia', 'Linda Rodriguez', 'William Lee', 'Barbara Anderson',
        'John Taylor', 'Mary Thomas', 'Richard Jackson', 'Patricia White', 'Thomas Harris'
    ], n),
    'Department': np.random.choice(['Sales', 'Engineering', 'Product', 'Marketing', 'Finance', 'HR'], n),
    'Position': np.random.choice(['Junior', 'Senior', 'Lead', 'Manager', 'Director'], n),
    'Salary': np.random.uniform(50000, 200000, n).astype(int),
    'Age': np.random.uniform(22, 65, n).astype(int),
    'Tenure_Years': np.random.uniform(0.5, 20, n).round(1),
    'YearsSincePromotion': np.random.uniform(0, 10, n).round(1),
    'Performance_Rating': np.random.uniform(2.5, 5.0, n).round(1),
    'Training_Hours': np.random.uniform(10, 100, n).astype(int),
    'Attrition_Risk': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.5, 0.3, 0.2]),
    'Remote_Work': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
}

df = pd.DataFrame(data)

# Save to project folder
output_path = 'data/employees.csv'
df.to_csv(output_path, index=False)

print(f"✅ Dataset created: {output_path}")
print(f"📊 Records: {len(df)}")
print(f"📁 Location: {output_path}")
