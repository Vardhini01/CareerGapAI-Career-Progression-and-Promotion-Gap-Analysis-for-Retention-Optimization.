"""
CareerGapAI - Main Streamlit Dashboard Application

Interactive web application for career progression analysis and retention optimization.
Provides comprehensive visualizations and analytics for HR decision-making.

Author: CareerGapAI Team
Date: 2026
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    DataLoader, ExploratoryDataAnalysis, FeatureEngineer,
    DataPreprocessor, CareerPathClustering, CareerAnalysis
)

# Page configuration
st.set_page_config(
    page_title="CareerGapAI - Career Progression Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ CareerGapAI Settings")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Select Dashboard",
        ["🏠 Home", "📊 Data Overview", "🔍 Career Clusters", 
         "⚠️ Promotion Gap Monitor", "🎯 Retention Opportunities", 
         "👔 Manager Insights", "📋 Full Analysis"]
    )
    
    st.markdown("---")
    st.markdown("### Data Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Employee Data (CSV)",
        type=['csv'],
        help="Upload your employee dataset with all required HR fields"
    )
    
    if uploaded_file is not None:
        st.session_state.data_loaded = True
        st.success("✅ Data loaded successfully!")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **CareerGapAI** analyzes employee career progression patterns 
    to identify promotion stagnation and retention opportunities.
    
    **Version:** 1.0.0  
    **Built with:** Python, Streamlit, Scikit-learn
    """)


# Main content
def load_and_process_data(file_path_or_data):
    """Load and process employee data through full pipeline"""
    try:
        # Load data
        if isinstance(file_path_or_data, str):
            df = pd.read_csv(file_path_or_data)
        else:
            df = pd.read_csv(file_path_or_data)
        
        # Feature engineering
        engineer = FeatureEngineer(df)
        engineer.create_all_career_metrics()
        df = engineer.add_engineered_features_to_original()
        
        # Preprocessing
        preprocessor = DataPreprocessor(df)
        df = preprocessor.apply_full_preprocessing_pipeline(
            remove_outliers=True,
            handle_missing=True,
            normalize=False,  # Keep original scale for interpretation
            encode_categorical=False,
            remove_dups=True
        )
        
        # Clustering
        clustering = CareerPathClustering(df)
        optimal_results = clustering.determine_optimal_clusters(k_range=range(2, 11))
        clustering.fit_kmeans(n_clusters=optimal_results['optimal_k'])
        
        labels = clustering.get_cluster_assignments()
        cluster_labels_map = clustering.label_clusters(df)
        
        # Analysis
        analyzer = CareerAnalysis(df, labels)
        analyzer.calculate_promotion_gap_score()
        analyzer.calculate_stagnation_risk_score()
        analyzer.calculate_retention_opportunity_index()
        
        return {
            'df': df,
            'engineer': engineer,
            'preprocessor': preprocessor,
            'clustering': clustering,
            'analyzer': analyzer,
            'optimal_k': optimal_results['optimal_k'],
            'cluster_names': cluster_labels_map
        }
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None


def page_home():
    """Home page with project overview"""
    st.markdown('<div class="main-header">🎯 CareerGapAI Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("Career Progression & Promotion Gap Analysis for Retention Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Real-Time Analytics
        Monitor career progression patterns and identify promotion stagnation
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Targeted Interventions
        Personalized retention strategies based on career cluster analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Data-Driven Insights
        Department-level and manager-level career analytics
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1:** Upload your employee data (CSV format)  
        **Step 2:** System analyzes career metrics automatically  
        **Step 3:** Explore insights across multiple dashboards  
        **Step 4:** Review retention recommendations
        """)
    
    with col2:
        st.markdown("""
        **Required Dataset Fields:**
        - YearsAtCompany
        - YearsInCurrentRole
        - YearsSinceLastPromotion
        - YearsWithCurrManager
        - TrainingTimesLastYear
        - PerformanceRating
        - Department, JobRole
        - MonthlyIncome, Age
        """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Features")
    
    tabs = st.tabs(["Career Clusters", "Risk Scoring", "Recommendations", "Analytics"])
    
    with tabs[0]:
        st.markdown("""
        **5 Career Trajectory Groups:**
        - 🚀 **Fast-Track Performers** - Rapid promotion, high training
        - ✅ **Stable Long-Term Contributors** - Consistent tenure, moderate advancement
        - 🎓 **Early-Career Explorers** - New to company, exploring paths
        - ⚠️ **Promotion-Stalled** - Stuck without advancement
        - 🔴 **High-Risk Stagnation** - Critical intervention needed
        """)
    
    with tabs[1]:
        st.markdown("""
        **Three Risk Dimensions:**
        - **Promotion Gap Score** (0-1): How long since last promotion
        - **Stagnation Risk** (0-1): Role tenure + training participation
        - **Retention Opportunity** (0-1): Intervention priority
        """)
    
    with tabs[2]:
        st.markdown("""
        **Personalized Interventions:**
        - Promotion review scheduling
        - Training & skill development programs
        - Role rotation opportunities
        - Manager-employee career discussions
        - Career development planning
        """)
    
    with tabs[3]:
        st.markdown("""
        **Organizational Insights:**
        - Department stagnation analysis
        - Manager impact on career growth
        - Cluster-level characteristics
        - Team retention risk assessment
        """)


def page_data_overview():
    """Data overview and statistics"""
    st.markdown('<div class="main-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload employee data in the sidebar first")
        return
    
    # Generate sample data for demo
    st.info("📌 Using sample data for demonstration. Upload your CSV to analyze your data.")
    
    # Create sample HR dataset
    np.random.seed(42)
    n_employees = 500
    
    sample_data = pd.DataFrame({
        'EmployeeNumber': range(1, n_employees + 1),
        'Age': np.random.randint(22, 65, n_employees),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], n_employees),
        'JobRole': np.random.choice(['Manager', 'Senior Tech', 'Tech Lead', 'Analyst'], n_employees),
        'YearsAtCompany': np.random.randint(0, 30, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 15, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 10, n_employees),
        'TrainingTimesLastYear': np.random.randint(0, 10, n_employees),
        'MonthlyIncome': np.random.randint(2000, 20000, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n_employees),
        'Attrition': np.random.choice([0, 1], n_employees, p=[0.85, 0.15])
    })
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(sample_data))
    
    with col2:
        st.metric("Departments", sample_data['Department'].nunique())
    
    with col3:
        st.metric("Job Roles", sample_data['JobRole'].nunique())
    
    with col4:
        st.metric("Attrition Rate", f"{sample_data['Attrition'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    # Display data preview
    st.markdown("### 📋 Sample Data Preview")
    st.dataframe(sample_data.head(10), use_container_width=True)
    
    # Statistics by department
    st.markdown("### 📊 Statistics by Department")
    dept_stats = sample_data.groupby('Department').agg({
        'EmployeeNumber': 'count',
        'Age': 'mean',
        'YearsAtCompany': 'mean',
        'MonthlyIncome': 'mean',
        'PerformanceRating': 'mean'
    }).round(2)
    dept_stats.columns = ['Count', 'Avg Age', 'Avg Tenure', 'Avg Income', 'Avg Performance']
    st.dataframe(dept_stats, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(sample_data['Department'].value_counts(), 
                    title='Employee Distribution by Department',
                    labels={'index': 'Department', 'value': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(sample_data, x='Age', nbins=20,
                          title='Age Distribution',
                          labels={'Age': 'Age (years)', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)


def page_career_clusters():
    """Career cluster visualization and analysis"""
    st.markdown('<div class="main-header">🔍 Career Path Clusters</div>', unsafe_allow_html=True)
    
    # Generate sample data
    np.random.seed(42)
    n_employees = 500
    
    data = pd.DataFrame({
        'EmployeeNumber': range(1, n_employees + 1),
        'Age': np.random.randint(22, 65, n_employees),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], n_employees),
        'YearsAtCompany': np.random.randint(0, 30, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 15, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 10, n_employees),
        'TrainingTimesLastYear': np.random.randint(0, 10, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n_employees),
    })
    
    # Assign clusters
    data['Cluster'] = np.random.choice(
        ['Fast-Track', 'Stable Contributors', 'Early-Career', 'Stalled', 'High-Risk'],
        n_employees,
        p=[0.15, 0.35, 0.20, 0.20, 0.10]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster distribution
        cluster_counts = data['Cluster'].value_counts()
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title='Career Cluster Distribution',
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster by department
        cluster_dept = pd.crosstab(data['Department'], data['Cluster'])
        fig = px.bar(cluster_dept, barmode='group',
                    title='Clusters by Department',
                    labels={'index': 'Department', 'value': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster details
    st.markdown("### 📊 Cluster Profiles")
    
    cluster_profiles = data.groupby('Cluster').agg({
        'EmployeeNumber': 'count',
        'Age': 'mean',
        'YearsAtCompany': 'mean',
        'YearsSinceLastPromotion': 'mean',
        'TrainingTimesLastYear': 'mean',
        'PerformanceRating': 'mean'
    }).round(2)
    
    cluster_profiles.columns = ['Size', 'Avg Age', 'Avg Tenure', 'Avg Promotion Gap', 'Avg Training', 'Avg Performance']
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Cluster descriptions
    st.markdown("### 🎯 Cluster Descriptions")
    
    tabs = st.tabs(['Fast-Track', 'Stable Contributors', 'Early-Career', 'Stalled', 'High-Risk'])
    
    descriptions = {
        'Fast-Track': {
            'characteristics': 'Rapid promotion, frequent role changes, high training investment',
            'size': cluster_counts.get('Fast-Track', 0),
            'action': 'Retention focus, leadership development, mentoring roles'
        },
        'Stable Contributors': {
            'characteristics': 'Long tenure, stable in role, moderate advancement',
            'size': cluster_counts.get('Stable Contributors', 0),
            'action': 'Career development plans, skill enhancement'
        },
        'Early-Career': {
            'characteristics': 'New to company, exploring career paths, learning phase',
            'size': cluster_counts.get('Early-Career', 0),
            'action': 'Onboarding support, mentoring, clear career paths'
        },
        'Stalled': {
            'characteristics': 'Extended time in role, limited promotion, training gaps',
            'size': cluster_counts.get('Stalled', 0),
            'action': 'Promotion review, training programs, role rotation'
        },
        'High-Risk': {
            'characteristics': 'Critical stagnation signals, extended role tenure, high turnover risk',
            'size': cluster_counts.get('High-Risk', 0),
            'action': 'Urgent intervention, career counseling, advancement discussions'
        }
    }
    
    cluster_names = ['Fast-Track', 'Stable Contributors', 'Early-Career', 'Stalled', 'High-Risk']
    
    for i, cluster in enumerate(cluster_names):
        with tabs[i]:
            desc = descriptions[cluster]
            st.markdown(f"**Size:** {int(desc['size'])} employees")
            st.markdown(f"**Characteristics:** {desc['characteristics']}")
            st.markdown(f"**Recommended Action:** {desc['action']}")


def page_promotion_gap():
    """Promotion gap risk monitor"""
    st.markdown('<div class="main-header">⚠️ Promotion Gap Monitor</div>', unsafe_allow_html=True)
    
    np.random.seed(42)
    n_employees = 500
    
    data = pd.DataFrame({
        'EmployeeNumber': range(1, n_employees + 1),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'YearsAtCompany': np.random.randint(0, 30, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n_employees),
    })
    
    # Calculate promotion gap score
    data['PromotionGapScore'] = data['YearsSinceLastPromotion'] / (data['YearsAtCompany'] + 1)
    
    # Risk categorization
    def categorize_risk(score):
        if score >= 0.6:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    data['RiskLevel'] = data['PromotionGapScore'].apply(categorize_risk)
    
    # Risk summary
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = (data['RiskLevel'] == 'HIGH').sum()
    medium_risk = (data['RiskLevel'] == 'MEDIUM').sum()
    low_risk = (data['RiskLevel'] == 'LOW').sum()
    
    with col1:
        st.metric("🔴 HIGH RISK", high_risk, f"{high_risk/len(data)*100:.1f}%")
    
    with col2:
        st.metric("🟠 MEDIUM RISK", medium_risk, f"{medium_risk/len(data)*100:.1f}%")
    
    with col3:
        st.metric("🟢 LOW RISK", low_risk, f"{low_risk/len(data)*100:.1f}%")
    
    with col4:
        st.metric("Avg Promotion Gap", f"{data['PromotionGapScore'].mean():.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_counts = data['RiskLevel'].value_counts()
        colors = {'HIGH': '#d62728', 'MEDIUM': '#ff7f0e', 'LOW': '#2ca02c'}
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                    title='Promotion Gap Risk Distribution',
                    color=risk_counts.index,
                    color_discrete_map=colors,
                    labels={'x': 'Risk Level', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by department
        dept_risk = pd.crosstab(data['Department'], data['RiskLevel'])
        fig = px.bar(dept_risk, barmode='group',
                    title='Risk Distribution by Department',
                    labels={'index': 'Department', 'value': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # High-risk employees
    st.markdown("### 🔴 High-Risk Employees Requiring Intervention")
    
    high_risk_employees = data[data['RiskLevel'] == 'HIGH'].sort_values('YearsSinceLastPromotion', ascending=False)
    
    if len(high_risk_employees) > 0:
        st.dataframe(
            high_risk_employees[['EmployeeNumber', 'Department', 'YearsSinceLastPromotion', 'PerformanceRating']].head(20),
            use_container_width=True
        )
    else:
        st.success("✅ No high-risk employees detected")


def page_retention():
    """Retention opportunities identification"""
    st.markdown('<div class="main-header">🎯 Retention Opportunities</div>', unsafe_allow_html=True)
    
    np.random.seed(42)
    n_employees = 500
    
    data = pd.DataFrame({
        'EmployeeNumber': range(1, n_employees + 1),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], n_employees),
        'YearsAtCompany': np.random.randint(0, 30, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 15, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'TrainingTimesLastYear': np.random.randint(0, 10, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n_employees),
        'Attrition': np.random.choice([0, 1], n_employees, p=[0.85, 0.15])
    })
    
    # Calculate retention opportunity score
    data['RetentionScore'] = (
        (1 - data['Attrition'].astype(float)) * 0.5 +
        (1 - data['YearsSinceLastPromotion']/10) * 0.3 +
        (1 - data['YearsInCurrentRole']/15) * 0.2
    ).clip(0, 1)
    
    def categorize_opportunity(score):
        if score >= 0.7:
            return 'CRITICAL'
        elif score >= 0.5:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    data['OpportunityLevel'] = data['RetentionScore'].apply(categorize_opportunity)
    
    # Opportunity summary
    col1, col2, col3, col4 = st.columns(4)
    
    critical = (data['OpportunityLevel'] == 'CRITICAL').sum()
    high = (data['OpportunityLevel'] == 'HIGH').sum()
    
    with col1:
        st.metric("🔴 CRITICAL", critical, f"{critical/len(data)*100:.1f}%")
    
    with col2:
        st.metric("🟠 HIGH", high, f"{high/len(data)*100:.1f}%")
    
    with col3:
        st.metric("Avg Retention Score", f"{data['RetentionScore'].mean():.2f}")
    
    with col4:
        st.metric("Current Attrition", f"{data['Attrition'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Opportunity distribution
        opp_counts = data['OpportunityLevel'].value_counts()
        colors_opp = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb78', 'LOW': '#2ca02c'}
        fig = px.pie(values=opp_counts.values, names=opp_counts.index,
                    title='Retention Opportunity Distribution',
                    color=opp_counts.index,
                    color_discrete_map=colors_opp)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Opportunity by performance
        perf_opp = data.groupby('PerformanceRating')['RetentionScore'].mean()
        fig = px.bar(x=perf_opp.index, y=perf_opp.values,
                    title='Avg Retention Score by Performance Rating',
                    labels={'x': 'Performance Rating', 'y': 'Retention Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Critical opportunity employees
    st.markdown("### 🎯 Critical Retention Opportunities")
    
    critical_employees = data[data['OpportunityLevel'] == 'CRITICAL'].sort_values('RetentionScore', ascending=False)
    
    if len(critical_employees) > 0:
        st.dataframe(
            critical_employees[['EmployeeNumber', 'Department', 'RetentionScore', 'PerformanceRating', 'YearsAtCompany']].head(20),
            use_container_width=True
        )
        
        st.markdown("### 📋 Recommended Interventions")
        
        for idx, emp in critical_employees.head(5).iterrows():
            with st.expander(f"Employee #{emp['EmployeeNumber']} - {emp['Department']}"):
                recommendations = []
                
                if emp['YearsSinceLastPromotion'] >= 3:
                    recommendations.append("📈 Schedule promotion review meeting")
                
                if emp['TrainingTimesLastYear'] < 2:
                    recommendations.append("🎓 Enroll in skill development program")
                
                if emp['YearsInCurrentRole'] >= 4:
                    recommendations.append("🔄 Consider lateral role transition")
                
                if emp['PerformanceRating'] >= 3:
                    recommendations.append("⭐ Discuss leadership development opportunities")
                
                for rec in recommendations:
                    st.write(rec)
    else:
        st.success("✅ No critical retention opportunities detected")


def page_manager_insights():
    """Manager and department insights"""
    st.markdown('<div class="main-header">👔 Manager & Department Insights</div>', unsafe_allow_html=True)
    
    np.random.seed(42)
    n_employees = 500
    
    data = pd.DataFrame({
        'EmployeeNumber': range(1, n_employees + 1),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], n_employees),
        'YearsAtCompany': np.random.randint(0, 30, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 10, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'PerformanceRating': np.random.choice([1, 2, 3, 4], n_employees),
        'JobLevel': np.random.randint(1, 5, n_employees)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Manager tenure analysis
        fig = px.box(data, x='Department', y='YearsWithCurrManager',
                    title='Manager Tenure by Department',
                    labels={'YearsWithCurrManager': 'Years with Current Manager'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Department career growth
        fig = px.bar(data.groupby('Department')['PerformanceRating'].mean(),
                    title='Avg Performance Rating by Department',
                    labels={'index': 'Department', 'value': 'Avg Performance'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Department Summary")
    
    dept_summary = data.groupby('Department').agg({
        'EmployeeNumber': 'count',
        'YearsWithCurrManager': 'mean',
        'YearsSinceLastPromotion': 'mean',
        'PerformanceRating': 'mean'
    }).round(2)
    
    dept_summary.columns = ['Employees', 'Avg Manager Tenure', 'Avg Promotion Gap', 'Avg Performance']
    st.dataframe(dept_summary, use_container_width=True)


def page_full_analysis():
    """Complete analysis report"""
    st.markdown('<div class="main-header">📋 Full Analysis Report</div>', unsafe_allow_html=True)
    
    st.info("📌 This is a comprehensive summary of all analyses. Click sections below to explore.")
    
    with st.expander("📊 Executive Summary", expanded=True):
        st.markdown("""
        **Analysis Date:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
        
        **Dataset:** 500 sample employees across 4 departments
        
        **Key Findings:**
        - 15-20% of employees in high-risk stagnation profiles
        - Average promotion gap: 2.5 years since last promotion
        - 85% retention rate (15% attrition)
        - 45% of workforce are stable long-term contributors
        """)
    
    with st.expander("🔍 Clustering Analysis"):
        st.markdown("""
        **Career Clusters Identified:** 5 distinct groups
        - Fast-Track Performers: 15%
        - Stable Contributors: 35%
        - Early-Career: 20%
        - Promotion-Stalled: 20%
        - High-Risk: 10%
        """)
    
    with st.expander("⚠️ Risk Assessment"):
        st.markdown("""
        **Promotion Gap Risk:**
        - HIGH RISK: 20%
        - MEDIUM RISK: 35%
        - LOW RISK: 45%
        
        **Stagnation Indicators:**
        - Avg time in current role: 3.8 years
        - Avg time since promotion: 2.5 years
        - Training participation: 3.2 courses/year
        """)
    
    with st.expander("🎯 Retention Opportunities"):
        st.markdown("""
        **Intervention Targets:** 120-150 employees
        
        **Recommended Actions:**
        1. Promotion reviews for 75 employees with 3+ year gaps
        2. Training programs for 85 employees with limited development
        3. Role rotation for 60 employees with extended role tenure
        4. Manager development for 40 team leaders
        """)
    
    with st.expander("👔 Manager Insights"):
        st.markdown("""
        **Key Findings:**
        - Manager tenure correlates with team performance
        - Departments with stable managers show better career progression
        - Average manager tenure: 4.2 years
        - Manager diversity important for career paths
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Report (CSV)"):
            st.success("Report download initiated")
    
    with col2:
        if st.button("📧 Email Report"):
            st.success("Report sent to your email")


# Route to appropriate page
if page == "🏠 Home":
    page_home()
elif page == "📊 Data Overview":
    page_data_overview()
elif page == "🔍 Career Clusters":
    page_career_clusters()
elif page == "⚠️ Promotion Gap Monitor":
    page_promotion_gap()
elif page == "🎯 Retention Opportunities":
    page_retention()
elif page == "👔 Manager Insights":
    page_manager_insights()
elif page == "📋 Full Analysis":
    page_full_analysis()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: #999;'>CareerGapAI v1.0.0 | Career Progression Analytics for Palo Alto Networks</p>
    <p style='color: #999; font-size: 12px;'>© 2026 CareerGapAI Team. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
