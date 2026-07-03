"""
CareerGapAI - Modern Dashboard Application
Enterprise-grade Career Progression Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    DataLoader, ExploratoryDataAnalysis, FeatureEngineer,
    DataPreprocessor, CareerPathClustering, CareerAnalysis
)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CareerGapAI - Enterprise Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== COLOR PALETTE ====================
COLORS = {
    'primary': '#1F77B4',
    'secondary': '#FF7F0E',
    'success': '#2CA02C',
    'danger': '#D62728',
    'warning': '#FFA500',
    'dark': '#0F0F0F',
    'light': '#F0F2F6',
    'accent': '#00D9FF',
    'text_dark': '#1E1E1E',
    'text_light': '#FFFFFF',
    'subtle': '#E8EAED'
}

# ==================== CUSTOM CSS ====================
st.markdown(f"""
<style>
    /* Global */
    :root {{
        --primary: {COLORS['primary']};
        --secondary: {COLORS['secondary']};
        --success: {COLORS['success']};
        --danger: {COLORS['danger']};
    }}
    
    * {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 0;
        padding: 0;
    }}
    
    /* Main Background */
    .main {{
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 100%);
        color: #E8EAED;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0F0F0F 0%, #1A1A1E 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Headers */
    h1 {{
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00D9FF 0%, #1F77B4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
    }}
    
    h2 {{
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #E8EAED !important;
        border-bottom: 2px solid rgba(31, 119, 180, 0.3);
        padding-bottom: 10px !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }}
    
    h3 {{
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #00D9FF !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: rgba(31, 119, 180, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 119, 180, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .metric-card:hover {{
        border: 1px solid rgba(0, 217, 255, 0.5);
        box-shadow: 0 12px 48px rgba(0, 217, 255, 0.25);
        transform: translateY(-4px);
        background: rgba(31, 119, 180, 0.15);
    }}
    
    /* Highlight Text */
    .highlight {{
        background: linear-gradient(120deg, #FF7F0E 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    .highlight-danger {{
        background: linear-gradient(120deg, #D62728 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    .highlight-success {{
        background: linear-gradient(120deg, #2CA02C 0%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    /* KPI Display */
    .kpi-container {{
        display: flex;
        gap: 15px;
        margin: 15px 0;
        flex-wrap: wrap;
    }}
    
    .kpi-box {{
        flex: 1;
        min-width: 200px;
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.15) 0%, rgba(0, 217, 255, 0.05) 100%);
        border-left: 4px solid #00D9FF;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .kpi-label {{
        font-size: 0.85rem;
        color: #A8B2D1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #00D9FF;
        margin-top: 8px;
    }}
    
    /* Badge/Tags */
    .badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .badge-primary {{ 
        background: {COLORS['primary']}; 
        color: white;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }}
    
    .badge-danger {{ 
        background: {COLORS['danger']}; 
        color: white;
        box-shadow: 0 4px 12px rgba(214, 39, 40, 0.3);
    }}
    
    .badge-success {{ 
        background: {COLORS['success']}; 
        color: white;
        box-shadow: 0 4px 12px rgba(44, 160, 44, 0.3);
    }}
    
    .badge-warning {{ 
        background: {COLORS['warning']}; 
        color: white;
        box-shadow: 0 4px 12px rgba(255, 127, 14, 0.3);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3) !important;
        letter-spacing: 0.5px;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.5) !important;
        transform: translateY(-2px) !important;
    }}
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stFileUploader > div > div > button {{
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        color: #E8EAED !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {{
        border: 1px solid {COLORS['accent']} !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.2) !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        background: transparent;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']} !important;
        border-bottom: 3px solid {COLORS['accent']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: rgba(0, 217, 255, 0.1) !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
    }}
    
    /* Text */
    p, span, label {{
        color: #E8EAED !important;
    }}
    
    /* Radio buttons */
    [data-baseweb="radio"] label {{
        color: #E8EAED !important;
    }}
    
    /* Divider */
    hr {{
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# ==================== HELPER FUNCTIONS ====================
def create_metric_card(label, value, change=None, unit="", icon=""):
    """Create a styled metric card"""
    html = f"""
    <div class="metric-card">
        <p class="kpi-label">{icon} {label}</p>
        <p class="kpi-value">{value} <span style="font-size: 1rem; color: #A8B2D1;">{unit}</span></p>
        {f'<p style="font-size: 0.9rem; color: #00D9FF; margin-top: 8px;">{change}</p>' if change else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_kpi_grid(kpis):
    """Create KPI grid layout"""
    cols = st.columns(len(kpis))
    for col, (label, value, change, unit, icon) in zip(cols, kpis):
        with col:
            create_metric_card(label, value, change, unit, icon)

# ==================== SAMPLE DATA GENERATION ====================
@st.cache_resource
def generate_sample_data():
    """Generate sample employee data"""
    np.random.seed(42)
    
    # Load from CSV if exists
    csv_path = os.path.join(os.path.dirname(__file__), '../data/sample_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    
    # Fallback: generate sample data
    n_employees = 400
    departments = ['Sales', 'R&D', 'IT', 'Finance', 'HR']
    
    df = pd.DataFrame({
        'EmployeeID': [f'E{i:04d}' for i in range(1, n_employees + 1)],
        'Department': np.random.choice(departments, n_employees),
        'YearsAtCompany': np.random.exponential(5, n_employees) + 1,
        'YearsInCurrentRole': np.random.exponential(3, n_employees) + 0.5,
        'YearsSinceLastPromotion': np.random.exponential(3, n_employees),
        'PerformanceRating': np.random.uniform(3.0, 5.0, n_employees),
        'TrainingHoursPerYear': np.random.exponential(25, n_employees) + 10,
        'AnnualSalary': np.random.uniform(50000, 160000, n_employees),
        'Age': np.random.uniform(25, 65, n_employees),
        'YearsWithCurrentManager': np.random.exponential(3, n_employees) + 0.5
    })
    
    return df

# ==================== PROCESS DATA ====================
@st.cache_resource
def process_career_data(df):
    """Process career data through pipeline"""
    try:
        # Preprocessing
        preprocessor = DataPreprocessor(df)
        df_clean = preprocessor.apply_full_preprocessing_pipeline()
        
        # Feature Engineering
        engineer = FeatureEngineer(df_clean)
        df_features = engineer.create_all_features()
        
        # Clustering
        clusterer = CareerPathClustering(df_features)
        optimal_k = clusterer.determine_optimal_clusters(k_range=range(3, 8), method='silhouette')
        clusterer.fit_kmeans(optimal_k)
        df_clusters = clusterer.profile_clusters(df_features, method='kmeans')
        
        # Analysis
        analyzer = CareerAnalysis(df_clusters)
        
        return df_clusters, analyzer, clusterer, optimal_k
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None

# ==================== PAGE: HOME ====================
def page_home():
    st.markdown("# 📊 CareerGapAI")
    st.markdown("**Enterprise Career Progression Analytics Platform**")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Overview")
        st.markdown("""
        **CareerGapAI** is an advanced analytics platform that helps HR leaders and managers:
        
        - 🔍 **Identify Career Patterns** - Discover employee career progression trends
        - ⚠️ **Detect At-Risk Talent** - Find employees likely to leave
        - 📈 **Optimize Retention** - Implement data-driven interventions
        - 👥 **Benchmark Performance** - Compare across departments and managers
        - 💡 **Actionable Insights** - Get specific recommendations for each employee
        """)
    
    with col2:
        st.markdown("### ⚡ Key Features")
        st.markdown("""
        <div style="background: rgba(0, 217, 255, 0.1); border-left: 4px solid #00D9FF; padding: 15px; border-radius: 8px;">
        <p><strong>5</strong> Career Clusters</p>
        <p><strong>3</strong> Risk Dimensions</p>
        <p><strong>4</strong> Career Metrics</p>
        <p><strong>7</strong> Dashboard Pages</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    steps = st.columns(4)
    with steps[0]:
        st.markdown("<h4 style='text-align: center; color: #00D9FF;'>1️⃣ Upload</h4>", unsafe_allow_html=True)
        st.markdown("Upload your employee CSV")
    with steps[1]:
        st.markdown("<h4 style='text-align: center; color: #00D9FF;'>2️⃣ Analyze</h4>", unsafe_allow_html=True)
        st.markdown("System processes data")
    with steps[2]:
        st.markdown("<h4 style='text-align: center; color: #00D9FF;'>3️⃣ Explore</h4>", unsafe_allow_html=True)
        st.markdown("View visualizations")
    with steps[3]:
        st.markdown("<h4 style='text-align: center; color: #00D9FF;'>4️⃣ Act</h4>", unsafe_allow_html=True)
        st.markdown("Implement actions")

# ==================== PAGE: DATA OVERVIEW ====================
def page_data_overview(df):
    st.markdown("# 📊 Data Overview")
    st.markdown("Dataset statistics and distribution analysis")
    st.markdown("---")
    
    # Basic stats
    kpis = [
        ("Total Employees", f"{len(df):,}", None, "", "👥"),
        ("Departments", f"{df['Department'].nunique()}", None, "", "🏢"),
        ("Avg Salary", f"${df['AnnualSalary'].mean():,.0f}", None, "", "💰"),
        ("Avg Tenure", f"{df['YearsAtCompany'].mean():.1f}", "years", "", "📅"),
    ]
    create_kpi_grid(kpis)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Department Distribution")
        dept_counts = df['Department'].value_counts()
        fig = px.pie(
            values=dept_counts.values,
            names=dept_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.3
        )
        fig.update_layout(
            template="plotly_dark",
            font=dict(size=12, color="#E8EAED"),
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💼 Salary Distribution")
        fig = px.box(
            df,
            y='AnnualSalary',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.update_layout(
            template="plotly_dark",
            font=dict(size=12, color="#E8EAED"),
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Annual Salary ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📋 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

# ==================== PAGE: CAREER CLUSTERS ====================
def page_career_clusters(df, analyzer, clusterer):
    st.markdown("# 🔍 Career Clusters")
    st.markdown("Identify distinct career progression patterns")
    st.markdown("---")
    
    cluster_names = {
        0: "Fast-Track Performers",
        1: "Stable Contributors",
        2: "Early-Career Explorers",
        3: "Promotion-Stalled",
        4: "High-Risk Stagnation"
    }
    
    # Cluster stats
    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(5):
        with [col1, col2, col3, col4, col5][i]:
            count = (df['Cluster'] == i).sum() if 'Cluster' in df.columns else 0
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            st.metric(f"Cluster {i}", f"{count}", f"{pct:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Cluster Distribution")
        if 'Cluster' in df.columns:
            cluster_dist = df['Cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_dist.index.astype(str),
                y=cluster_dist.values,
                color=cluster_dist.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={'x': 'Cluster', 'y': 'Count'}
            )
            fig.update_layout(
                template="plotly_dark",
                font=dict(size=12, color="#E8EAED"),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 Cluster Profiles")
        if 'Cluster' in df.columns:
            cluster_summary = df.groupby('Cluster').agg({
                'AnnualSalary': 'mean',
                'YearsAtCompany': 'mean',
                'PerformanceRating': 'mean'
            }).round(2)
            st.dataframe(cluster_summary, use_container_width=True)

# ==================== PAGE: PROMOTION GAP MONITOR ====================
def page_promotion_gap(df, analyzer):
    st.markdown("# ⚠️ Promotion Gap Monitor")
    st.markdown("Identify employees at risk due to promotion delays")
    st.markdown("---")
    
    try:
        if analyzer:
            promo_scores = analyzer.calculate_promotion_gap_score()
            
            # Risk categorization
            high_risk = (promo_scores >= 0.6).sum()
            medium_risk = ((promo_scores >= 0.3) & (promo_scores < 0.6)).sum()
            low_risk = (promo_scores < 0.3).sum()
            
            # KPIs
            kpis = [
                ("High Risk", f"{high_risk}", f"{high_risk/len(df)*100:.1f}%", "", "🔴"),
                ("Medium Risk", f"{medium_risk}", f"{medium_risk/len(df)*100:.1f}%", "", "🟠"),
                ("Low Risk", f"{low_risk}", f"{low_risk/len(df)*100:.1f}%", "", "🟢"),
                ("Avg Gap Score", f"{promo_scores.mean():.2f}", "out of 1.0", "", "📊"),
            ]
            create_kpi_grid(kpis)
            
            st.markdown("---")
            
            col1, col2 = st.columns([1.2, 0.8])
            
            with col1:
                st.markdown("### 📈 Risk Distribution")
                risk_dist = pd.Series({
                    'High Risk': high_risk,
                    'Medium Risk': medium_risk,
                    'Low Risk': low_risk
                })
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    color_discrete_map={
                        'High Risk': COLORS['danger'],
                        'Medium Risk': COLORS['warning'],
                        'Low Risk': COLORS['success']
                    }
                )
                fig.update_layout(
                    template="plotly_dark",
                    font=dict(size=12, color="#E8EAED"),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Top At-Risk")
                risk_df = pd.DataFrame({
                    'Employee': df['EmployeeID'].head(5),
                    'Gap Score': promo_scores.head(5).round(2),
                    'Years': df['YearsSinceLastPromotion'].head(5).round(1)
                })
                st.dataframe(risk_df, use_container_width=True)
    except:
        st.info("Run full analysis to see promotion gap data")

# ==================== PAGE: RETENTION OPPORTUNITIES ====================
def page_retention(df, analyzer):
    st.markdown("# 🎯 Retention Opportunities")
    st.markdown("Identify and prioritize employees for intervention")
    st.markdown("---")
    
    try:
        if analyzer:
            retention_index = analyzer.calculate_retention_opportunity_index()
            
            critical = (retention_index >= 0.7).sum()
            high = ((retention_index >= 0.5) & (retention_index < 0.7)).sum()
            medium = ((retention_index >= 0.3) & (retention_index < 0.5)).sum()
            
            kpis = [
                ("Critical", f"{critical}", f"{critical/len(df)*100:.1f}%", "🔴"),
                ("High", f"{high}", f"{high/len(df)*100:.1f}%", "🟠"),
                ("Medium", f"{medium}", f"{medium/len(df)*100:.1f}%", "🟡"),
                ("Avg Retention Index", f"{retention_index.mean():.2f}", "", "📊"),
            ]
            
            cols = st.columns(len(kpis))
            for col, (label, value, change, icon) in zip(cols, kpis):
                with col:
                    create_metric_card(label, value, change, "", icon)
            
            st.markdown("---")
            
            st.markdown("### 🚨 Critical Attention Needed")
            critical_employees = df[retention_index >= 0.7].head(10)
            if len(critical_employees) > 0:
                display_df = critical_employees[['EmployeeID', 'Department', 'YearsAtCompany', 'PerformanceRating']].copy()
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No critical cases found")
    except:
        st.info("Run full analysis to see retention data")

# ==================== PAGE: MANAGER INSIGHTS ====================
def page_manager_insights(df):
    st.markdown("# 👔 Manager Insights")
    st.markdown("Manager effectiveness and team dynamics analysis")
    st.markdown("---")
    
    if 'YearsWithCurrentManager' in df.columns:
        manager_avg_tenure = df.groupby('Department')['YearsWithCurrentManager'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Avg Manager Tenure by Department")
            fig = px.bar(
                x=manager_avg_tenure.index,
                y=manager_avg_tenure.values,
                color=manager_avg_tenure.values,
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                template="plotly_dark",
                font=dict(size=12, color="#E8EAED"),
                xaxis_title="Department",
                yaxis_title="Years",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 👥 Department Performance")
            dept_perf = df.groupby('Department')['PerformanceRating'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=dept_perf.index,
                y=dept_perf.values,
                color_discrete_sequence=[COLORS['primary']]
            )
            fig.update_layout(
                template="plotly_dark",
                font=dict(size=12, color="#E8EAED"),
                xaxis_title="Department",
                yaxis_title="Avg Rating",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FULL ANALYSIS ====================
def page_full_analysis(df, analyzer, clusterer):
    st.markdown("# 📋 Full Analysis")
    st.markdown("Comprehensive career progression report")
    st.markdown("---")
    
    with st.expander("📊 **Dataset Overview**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", len(df))
        col2.metric("Departments", df['Department'].nunique())
        col3.metric("Avg Tenure", f"{df['YearsAtCompany'].mean():.1f}y")
        col4.metric("Avg Salary", f"${df['AnnualSalary'].mean()/1000:.0f}K")
    
    with st.expander("🔍 **Career Clusters**"):
        if 'Cluster' in df.columns:
            for cluster_id in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster_id]
                st.markdown(f"#### Cluster {cluster_id} ({len(cluster_data)} employees)")
                st.write(f"Avg Salary: ${cluster_data['AnnualSalary'].mean():,.0f}")
                st.write(f"Avg Tenure: {cluster_data['YearsAtCompany'].mean():.1f} years")
    
    with st.expander("⚠️ **Risk Summary**"):
        if analyzer:
            st.markdown("Risk assessment across all dimensions")
    
    with st.expander("💼 **Department Breakdown**"):
        dept_summary = df.groupby('Department').agg({
            'EmployeeID': 'count',
            'AnnualSalary': 'mean',
            'PerformanceRating': 'mean',
            'YearsAtCompany': 'mean'
        }).round(2)
        dept_summary.columns = ['Count', 'Avg Salary', 'Avg Rating', 'Avg Tenure']
        st.dataframe(dept_summary, use_container_width=True)

# ==================== MAIN APP ====================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Settings")
        st.markdown("---")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📊 Data Overview", "🔍 Career Clusters",
             "⚠️ Promotion Gap", "🎯 Retention", "👔 Manager Insights",
             "📋 Full Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data Management")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if st.button("📊 Load Sample Data"):
            st.session_state.data_loaded = True
    
    # Load data
    if st.session_state.data_loaded or uploaded_file:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = generate_sample_data()
        
        df_processed, analyzer, clusterer, optimal_k = process_career_data(df)
        
        if df_processed is not None:
            st.session_state.df_processed = df_processed
        else:
            df_processed = df
        
        # Route pages
        if page == "🏠 Home":
            page_home()
        elif page == "📊 Data Overview":
            page_data_overview(df_processed)
        elif page == "🔍 Career Clusters":
            page_career_clusters(df_processed, analyzer, clusterer)
        elif page == "⚠️ Promotion Gap":
            page_promotion_gap(df_processed, analyzer)
        elif page == "🎯 Retention":
            page_retention(df_processed, analyzer)
        elif page == "👔 Manager Insights":
            page_manager_insights(df_processed)
        elif page == "📋 Full Analysis":
            page_full_analysis(df_processed, analyzer, clusterer)
    else:
        # Welcome screen
        st.markdown("# 📊 CareerGapAI")
        st.markdown("### Enterprise Career Progression Analytics")
        st.markdown("---")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("""
            Welcome to **CareerGapAI**, your comprehensive platform for:
            
            ✅ Analyzing career progression patterns  
            ✅ Identifying at-risk employees  
            ✅ Optimizing retention strategies  
            ✅ Data-driven HR decisions  
            
            **Get Started:**
            1. Click **"Load Sample Data"** in the sidebar (try it first!)
            2. Or upload your own employee CSV file
            3. Explore insights across 7 dashboard pages
            """)
        
        with col2:
            st.info("""
            **Features:**
            - 5 Career Clusters
            - 3 Risk Dimensions
            - 4 ML Metrics
            - 7 Dashboard Pages
            - Real-time Analysis
            """)
        
        st.markdown("---")
        st.markdown("**📊 Load sample data using the sidebar to get started!**")

if __name__ == "__main__":
    main()
