# CareerGapAI - Career Progression & Promotion Gap Analysis

## 🎯 Project Overview

CareerGapAI is a comprehensive Machine Learning + HR Analytics solution designed to analyze employee career progression patterns, identify promotion stagnation, and detect retention opportunities for **Palo Alto Networks**.

Unlike traditional attrition prediction models that only forecast *who* might leave, CareerGapAI explains **why** employees may eventually disengage by uncovering career progression issues and stagnation patterns. This enables **proactive, career-centric retention strategies**.

---

## 📋 Problem Statement

Many employees leave organizations not due to immediate dissatisfaction, but because of:

- **Long gaps between promotions** (2+ years without career advancement)
- **Role stagnation** (stuck in the same position for extended periods)
- **Limited skill growth** (insufficient training or development opportunities)
- **Weak managerial continuity** (frequent manager changes affecting relationships)

**Current gaps in HR analytics:**
- No data-driven view of career progression patterns
- Inability to identify promotion stagnation proactively
- Lack of early signals for retention opportunities
- Generic retention actions instead of personalized career interventions

---

## 🚀 Key Features

### 1. **Career Metrics Engineering** 📊
Four domain-specific metrics capture career progression patterns:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **PromotionGapRatio** | YearsSinceLastPromotion ÷ YearsAtCompany | Identifies promotion delays relative to tenure |
| **RoleStagnationIndex** | Weighted combination of role tenure & promotion gap | Detects employees stuck in same role |
| **TrainingIntensityScore** | Annual training normalized by tenure | Measures skill development investment |
| **ManagerStabilityIndicator** | YearsWithCurrManager ÷ YearsAtCompany | Evaluates relationship stability |

### 2. **Unsupervised Learning Clustering** 🔍
Identifies five distinct career trajectory groups:

1. **Fast-Track Performers** - Rapid promotion, high training investment
2. **Stable Long-Term Contributors** - Consistent tenure, moderate advancement
3. **Early-Career Explorers** - New to company, exploring career paths
4. **Promotion-Stalled Employees** - Stuck without advancement
5. **High-Risk Stagnation Profiles** - Critical intervention needed

### 3. **Risk Scoring & Intervention** 📈
Three risk dimensions guide retention actions:

- **Promotion Gap Risk Score** (0-1): How long since last promotion
- **Stagnation Risk Score** (0-1): Role tenure + training participation
- **Retention Opportunity Index** (0-1): Intervention priority for at-risk employees

### 4. **Intelligent Analysis** 🧠
- **Department-level insights**: Which departments have highest stagnation
- **Manager impact analysis**: How manager tenure affects career growth
- **Personalized recommendations**: Targeted interventions (promotion review, training, rotation)

### 5. **Interactive Streamlit Dashboard** 📱
Real-time visualization and exploration of:
- Career cluster distributions
- Promotion gap monitors
- Retention opportunity panels
- Department & manager insights
- Interactive filters and analytics

---

## 📁 Project Structure

```
CareerGapAI/
│
├── data/                                    # Raw & processed datasets
│   ├── raw_data.csv                        # Original employee data
│   └── processed_data.csv                  # Cleaned, engineered features
│
├── notebooks/                              # Jupyter notebooks
│   ├── 01_EDA.ipynb                       # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb       # Feature creation walkthrough
│   ├── 03_Clustering_Analysis.ipynb       # Clustering & profiling
│   └── 04_Full_Pipeline.ipynb             # End-to-end workflow
│
├── src/                                    # Core Python modules (production code)
│   ├── __init__.py                        # Package initialization
│   ├── data_loader.py                     # Data loading & validation
│   ├── eda.py                             # Exploratory data analysis
│   ├── feature_engineering.py             # Career metrics creation
│   ├── preprocessing.py                   # Data cleaning & transformation
│   ├── clustering.py                      # KMeans & Hierarchical clustering
│   └── analysis.py                        # Risk scoring & retention analysis
│
├── dashboard/                             # Streamlit application
│   ├── app.py                            # Main dashboard application
│   ├── pages/                            # Multi-page dashboard
│   │   ├── clustering_dashboard.py      # Career clusters visualization
│   │   ├── promotion_gap_monitor.py     # Promotion risk analysis
│   │   ├── retention_opportunities.py   # Intervention targets
│   │   └── manager_insights.py          # Manager-level analysis
│   └── assets/                          # Images, logos, styles
│
├── models/                               # Trained models & transformers
│   ├── kmeans_model.pkl                # Fitted KMeans model
│   ├── hierarchical_model.pkl          # Fitted Hierarchical model
│   ├── scalers/                        # Feature scalers
│   └── encoders/                       # Label encoders
│
├── outputs/                            # Analysis results & reports
│   ├── eda_report.txt                 # EDA findings
│   ├── cluster_profiles.csv           # Cluster characteristics
│   ├── risk_scores.csv                # Employee risk assessments
│   ├── retention_targets.csv          # Intervention target list
│   └── recommendations.json           # Personalized suggestions
│
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation (this file)
└── config.yaml                        # Configuration parameters

```

---

## 🔧 Tech Stack

**Data Processing & Analysis:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Statistical functions

**Machine Learning:**
- `scikit-learn` - Clustering (KMeans, Hierarchical)
- `scikit-optimize` - Hyperparameter optimization

**Visualization:**
- `matplotlib` - Static plots and charts
- `seaborn` - Statistical visualizations
- `plotly` - Interactive visualizations

**Dashboard & Web:**
- `streamlit` - Interactive web application framework
- `streamlit-option-menu` - Custom navigation

**Utilities:**
- `pydantic` - Data validation
- `python-dotenv` - Environment configuration

---

## 📊 Dataset Fields

The analysis uses 30+ employee fields including:

| Category | Fields |
|----------|--------|
| **Demographics** | Age, Gender, MaritalStatus, Education, EducationField |
| **Company Info** | Department, JobRole, JobLevel, DistanceFromHome |
| **Compensation** | MonthlyIncome, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel |
| **Career Metrics** | YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TotalWorkingYears |
| **Development** | TrainingTimesLastYear, Education, PerformanceRating |
| **Engagement** | JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, JobInvolvement, WorkLifeBalance |
| **Work Patterns** | OverTime, BusinessTravel, HourlyRate |
| **Outcome** | Attrition (0/1) |

---

## 📖 Module Documentation

### **src/data_loader.py** - Data Loading & Validation
```python
from src import DataLoader

loader = DataLoader('data')
df = loader.load_data('employees.csv')
loader.validate_required_columns(['Age', 'Department', 'YearsAtCompany'])
df_clean = loader.handle_missing_values(strategy='fill', numeric_strategy='median')
```

**Key Methods:**
- `load_data()` - Load CSV, Excel, or Parquet files
- `validate_required_columns()` - Check data completeness
- `handle_missing_values()` - Imputation strategies
- `remove_duplicates()` - Eliminate duplicate records
- `get_basic_statistics()` - Data profiling

---

### **src/eda.py** - Exploratory Data Analysis
```python
from src import ExploratoryDataAnalysis

eda = ExploratoryDataAnalysis(df, target_col='Attrition')
profile = eda.get_data_profile()
high_corr = eda.get_high_correlations(threshold=0.7)
eda.plot_correlation_heatmap()
report = eda.generate_eda_report()
```

**Key Methods:**
- `get_data_profile()` - Dataset overview
- `univariate_numeric_analysis()` - Numeric distribution analysis
- `univariate_categorical_analysis()` - Category frequency analysis
- `correlation_analysis()` - Correlation matrix
- `outlier_detection()` - IQR & Z-score methods
- `plot_numeric_distribution()` - Histogram & box plots
- `plot_correlation_heatmap()` - Visual correlation matrix

---

### **src/feature_engineering.py** - Career Metrics
```python
from src import FeatureEngineer

engineer = FeatureEngineer(df)
engineer.create_all_career_metrics()
df_engineered = engineer.add_engineered_features_to_original()
df_scaled = engineer.scale_numeric_features(method='minmax')
df_encoded = engineer.encode_categorical_features(method='auto')
```

**Key Methods:**
- `create_promotion_gap_ratio()` - Promotion delay metric
- `create_role_stagnation_index()` - Role tenure metric
- `create_training_intensity_score()` - Training investment metric
- `create_manager_stability_indicator()` - Manager continuity metric
- `scale_numeric_features()` - MinMax or Standard scaling
- `encode_categorical_features()` - Label or One-hot encoding

---

### **src/preprocessing.py** - Data Preparation
```python
from src import DataPreprocessor

preprocessor = DataPreprocessor(df)
preprocessor.apply_full_preprocessing_pipeline(
    remove_outliers=True,
    handle_missing=True,
    normalize=True,
    encode_categorical=True
)
df_processed = preprocessor.get_data()
summary = preprocessor.get_preprocessing_summary()
```

**Key Methods:**
- `remove_late_career_outliers()` - Domain-aware outlier removal
- `handle_missing_values_advanced()` - Strategic imputation
- `normalize_career_features()` - MinMax, Standard, Robust scaling
- `encode_categorical_features()` - Label or One-hot encoding
- `validate_data_quality()` - Quality checks

---

### **src/clustering.py** - Career Path Clustering
```python
from src import CareerPathClustering

clustering = CareerPathClustering(df_features)
results = clustering.determine_optimal_clusters(k_range=range(2, 11))
clustering.fit_kmeans(n_clusters=5)
clustering.fit_hierarchical(n_clusters=5)

comparison = clustering.compare_models()
labels = clustering.label_clusters(df)
clustering.plot_elbow_curve(results)
```

**Key Methods:**
- `determine_optimal_clusters()` - Elbow, Silhouette, combined methods
- `fit_kmeans()` - KMeans clustering
- `fit_hierarchical()` - Hierarchical clustering with multiple linkages
- `get_cluster_assignments()` - Retrieve cluster labels
- `profile_clusters()` - Cluster characteristics
- `label_clusters()` - Meaningful cluster naming
- `plot_elbow_curve()` - Visualization
- `plot_cluster_distribution()` - Distribution chart

---

### **src/analysis.py** - Risk Scoring & Retention Analysis
```python
from src import CareerAnalysis

analyzer = CareerAnalysis(df, cluster_labels)

# Risk scoring
promo_score = analyzer.calculate_promotion_gap_score()
stag_score = analyzer.calculate_stagnation_risk_score()
retention_score = analyzer.calculate_retention_opportunity_index()

# Identify targets
targets = analyzer.identify_retention_target_employees(threshold=0.5)

# Analysis
recommendations = analyzer.recommend_interventions()
dept_analysis = analyzer.analyze_by_department()
cluster_analysis = analyzer.analyze_by_cluster()

report = analyzer.generate_analysis_report()
```

**Key Methods:**
- `calculate_promotion_gap_score()` - Risk scoring (0-1)
- `calculate_stagnation_risk_score()` - Stagnation assessment
- `calculate_retention_opportunity_index()` - Intervention priority
- `identify_retention_target_employees()` - Target employee list
- `recommend_interventions()` - Personalized suggestions
- `analyze_by_department()` - Department insights
- `analyze_by_cluster()` - Cluster analysis
- `analyze_manager_impact()` - Manager influence on careers

---

## 🚀 Getting Started

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Your Data**
Place your employee dataset in the `data/` folder as `raw_data.csv` with all required columns.

### 3. **Run Full Pipeline**
```python
from src import DataLoader, ExploratoryDataAnalysis, FeatureEngineer
from src import DataPreprocessor, CareerPathClustering, CareerAnalysis

# Load data
loader = DataLoader('data')
df = loader.load_data('raw_data.csv')

# EDA
eda = ExploratoryDataAnalysis(df)
profile = eda.get_data_profile()

# Feature engineering
engineer = FeatureEngineer(df)
df = engineer.add_engineered_features_to_original()

# Preprocessing
preprocessor = DataPreprocessor(df)
df_processed = preprocessor.apply_full_preprocessing_pipeline()

# Clustering
clustering = CareerPathClustering(df_processed)
clustering.determine_optimal_clusters()
clustering.fit_kmeans(n_clusters=5)
labels = clustering.get_cluster_assignments()

# Analysis
analyzer = CareerAnalysis(df, labels)
targets = analyzer.identify_retention_target_employees()
report = analyzer.generate_analysis_report()
```

### 4. **Launch Interactive Dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📊 Key Performance Indicators (KPIs)

| KPI | Definition | Usage |
|-----|-----------|-------|
| **Career Cluster** | Employee career trajectory type (5 clusters) | Segmentation & targeting |
| **Promotion Gap Score** | 0-1 scale risk of promotion stagnation | Risk assessment |
| **Retention Opportunity Index** | 0-1 scale intervention priority | Targeting for interventions |
| **Stagnation Risk** | Combined role tenure + training participation | Early warning signal |
| **Manager Stability Impact** | How manager tenure affects career growth | Manager effectiveness |
| **Training Need Indicator** | Development gap identification | Training planning |

---

## 📈 Expected Outcomes

**For HR Teams:**
- Identify 15-25% of workforce requiring career interventions
- Personalized retention strategies vs. one-size-fits-all approaches
- Quantified ROI on career development programs

**For Managers:**
- Understand team career progression patterns
- Identify high-risk stagnation in direct reports
- Data-driven succession planning

**For Executives:**
- Proactive retention strategy reducing turnover costs
- Career-centric culture development
- Talent pipeline optimization

---

## 🎓 Methodology

This project implements an advanced data science methodology:

1. **Feature Engineering** - Create domain-specific career metrics
2. **Data Preprocessing** - Clean, normalize, and transform data
3. **Unsupervised Learning** - Identify career trajectory clusters
4. **Cluster Interpretation** - Label and characterize groups
5. **Risk Scoring** - Quantify stagnation and retention opportunities
6. **Intervention Design** - Generate actionable recommendations
7. **Dashboard Deployment** - Interactive analytics for stakeholders

---

## 📝 Output Files

| File | Purpose |
|------|---------|
| `eda_report.txt` | Statistical summary and insights |
| `cluster_profiles.csv` | Detailed cluster characteristics |
| `risk_scores.csv` | Employee-level risk assessments |
| `retention_targets.csv` | High-priority intervention targets |
| `recommendations.json` | Personalized career suggestions |

---

## 🤝 Contributing

This project is designed for educational and enterprise use. Contributions are welcome!

---

## 📞 Support & Contact

For questions or issues, please refer to the documentation in each module's docstrings.

---

## 📜 License

This project is proprietary and confidential.

---

## 🏢 About Palo Alto Networks

This analysis was designed specifically for Palo Alto Networks' HR analytics needs to improve career progression tracking and retention optimization.

---

**Version:** 1.0.0  
**Last Updated:** May 2026  
**Author:** CareerGapAI Team
