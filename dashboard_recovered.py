"""
CareerGapAI - Premium Analytics Dashboard
Modern, light-theme HR analytics platform for career progression analysis
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
    page_title="CareerGapAI - HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== LIGHT THEME COLOR PALETTE ====================
COLORS = {
    'primary': '#4F46E5',      # Indigo
    'secondary': '#06B6D4',    # Cyan
    'success': '#10B981',      # Emerald
    'danger': '#EF4444',       # Red
    'warning': '#F59E0B',      # Amber
    'bg': '#FFFFFF',           # White
    'bg_light': '#F9FAFB',     # Very light gray
    'bg_lighter': '#F3F4F6',   # Light gray
    'text': '#111827',         # Dark gray
    'text_light': '#6B7280',   # Light gray text
    'border': '#E5E7EB',       # Border gray
    'card_shadow': 'rgba(15, 23, 42, 0.08)',
}

# ==================== MODERN CSS STYLING ====================
st.markdown(f"""
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg_light']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        background-color: {COLORS['bg']};
    }}
    
    /* Headers */
    h1 {{
        color: {COLORS['text']} !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
    }}
    
    h2 {{
        color: {COLORS['text']} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid {COLORS['primary']} !important;
        padding-bottom: 0.75rem !important;
    }}
    
    h3 {{
        color: {COLORS['text']} !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }}
    
    /* Text */
    p, span, label {{
        color: {COLORS['text']} !important;
        font-size: 0.95rem !important;
        line-height: 1.6;
    }}
    
    /* Metric Cards Container */
    [data-testid="stMetricContainer"] {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, {COLORS['bg_light']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px {COLORS['card_shadow']};
        transition: all 0.3s ease;
    }}
    
    [data-testid="stMetricContainer"]:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.12);
        transform: translateY(-2px);
    }}
    
    /* KPI Value */
    [data-testid="stMetricDelta"] {{
        color: {COLORS['success']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35) !important;
    }}
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {{
        background-color: {COLORS['bg_lighter']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text']} !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    }}
    
    /* File Uploader */
    .stFileUploader {{
        background-color: {COLORS['bg_lighter']} !important;
        border: 2px dashed {COLORS['border']} !important;
        border-radius: 8px !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {COLORS['border']};
        background: transparent;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLORS['primary']} !important;
        border-bottom-color: {COLORS['primary']} !important;
        font-weight: 600 !important;
    }}
    
    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {COLORS['bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
    }}
    
    [data-testid="stExpander"] summary {{
        color: {COLORS['text']} !important;
        font-weight: 600 !important;
        background: transparent !important;
        padding: 1rem !important;
    }}
    
    /* Tables */
    .stDataFrame {{
        background-color: {COLORS['bg']} !important;
    }}
    
    .dataframe {{
        background-color: {COLORS['bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
    }}
    
    .stDataFrame table {{
        background-color: {COLORS['bg']} !important;
    }}
    
    /* Divider */
    hr {{
        border: none;
        border-top: 1px solid {COLORS['border']};
        margin: 2rem 0;
    }}
    
    /* Success/Info/Error Messages */
    .stAlert {{
        background-color: {COLORS['bg']} !important;
        border-left: 4px solid {COLORS['primary']} !important;
        border-radius: 8px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_sample_data():
    """Load sample HR analytics data"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '../data/hr_analytics.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    return None

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.0f}"

def format_percent(value):
    """Format value as percentage"""
    return f"{value*100:.1f}%"

def create_kpi_card(col, title, value, metric_format="text", change=None, icon=""):
    """Create a styled KPI card"""
    with col:
        with st.container():
            st.markdown(f"### {icon} {title}")
            if metric_format == "currency":
                st.markdown(f'<p style="font-size: 1.8rem; font-weight: 700; color: {COLORS["primary"]};">{format_currency(value)}</p>', unsafe_allow_html=True)
            elif metric_format == "percent":
                st.markdown(f'<p style="font-size: 1.8rem; font-weight: 700; color: {COLORS["secondary"]};">{format_percent(value)}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="font-size: 1.8rem; font-weight: 700; color: {COLORS["primary"]};">{value}</p>', unsafe_allow_html=True)
            
            if change:
                st.markdown(f'<p style="font-size: 0.85rem; color: {COLORS["text_light"]};">{change}</p>', unsafe_allow_html=True)

# ==================== PAGE: HOME ====================
def page_home():
    """Home/Landing page"""
    
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("# 📊 CareerGapAI")
        st.markdown("**Enterprise Career Progression Analytics Platform**")
    
    st.markdown("---")
    
    # Welcome section
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 🎯 Welcome")
        st.markdown("""
        CareerGapAI is your comprehensive HR analytics solution for:
        
        ✓ **Analyze Career Patterns** - Discover employee progression trends  
        ✓ **Identify At-Risk Talent** - Find high-departure risk employees  
        ✓ **Optimize Retention** - Data-driven intervention strategies  
        ✓ **Benchmark Performance** - Compare across teams and functions  
        ✓ **Drive HR Strategy** - Make informed, strategic decisions  
        """)
    
    with col2:
        st.markdown("### ⚡ Key Capabilities")
        with st.container():
            kpi_col1, kpi_col2 = st.columns(2)
            with kpi_col1:
                st.metric("Data Points", "1200+")
                st.metric("Departments", "7")
            with kpi_col2:
                st.metric("Metrics", "19+")
                st.metric("Analysis Types", "5")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("### 🚀 Getting Started")
    
    steps = st.columns(4)
    with steps[0]:
        st.markdown("**1️⃣ Load Data**")
        st.caption("Use sample or upload CSV")
    with steps[1]:
        st.markdown("**2️⃣ Process**")
        st.caption("Auto ML pipeline runs")
    with steps[2]:
        st.markdown("**3️⃣ Analyze**")
        st.caption("Explore insights")
    with steps[3]:
        st.markdown("**4️⃣ Act**")
        st.caption("Implement strategies")

# ==================== PAGE: DATA OVERVIEW ====================
def page_data_overview(df):
    """Data overview and exploration"""
    st.markdown("# 📈 Data Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    create_kpi_card(col1, "Total Employees", len(df), icon="👥")
    create_kpi_card(col2, "Departments", df['Department'].nunique(), icon="🏢")
    create_kpi_card(col3, "Avg Salary", df['AnnualSalary'].mean(), metric_format="currency", icon="💰")
    create_kpi_card(col4, "Avg Tenure", f"{df['YearsAtCompany'].mean():.1f} yrs", icon="📅")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### Department Distribution")
        dept_data = df['Department'].value_counts()
        fig = px.pie(
            values=dept_data.values,
            names=dept_data.index,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.3
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=11, color=COLORS['text']),
            showlegend=True,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Salary Distribution")
        fig = px.box(
            df,
            y='AnnualSalary',
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=11, color=COLORS['text']),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Annual Salary ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Employee Dataset")
    st.dataframe(
        df[['EmployeeID', 'FirstName', 'LastName', 'Department', 'Position', 'AnnualSalary', 'PerformanceRating']].head(20),
        use_container_width=True,
        hide_index=True
    )

# ==================== PAGE: CAREER CLUSTERS ====================
def page_career_clusters(df):
    """Career clustering analysis"""
    st.markdown("# 🔍 Career Clustering")
    st.markdown("Identify distinct employee career patterns")
    
    try:
        # Run clustering
        with st.spinner("Analyzing career patterns..."):
            clusterer = CareerPathClustering(df)
            optimal_k = clusterer.determine_optimal_clusters(k_range=range(3, 8), method='silhouette')
            clusterer.fit_kmeans(optimal_k)
            df_clusters = clusterer.profile_clusters(df, method='kmeans')
        
        # Cluster distribution
        col1, col2, col3, col4 = st.columns(4)
        for i in range(4):
            count = (df_clusters['Cluster'] == i).sum()
            pct = count / len(df_clusters) * 100
            create_kpi_card([col1, col2, col3, col4][i], f"Cluster {i}", f"{count} ({pct:.0f}%)")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### Cluster Distribution")
            cluster_counts = df_clusters['Cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                color=cluster_counts.index.astype(str),
                labels={'x': 'Cluster', 'y': 'Employees'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(
                template="plotly_white",
                font=dict(size=11, color=COLORS['text']),
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Avg Salary by Cluster")
            salary_by_cluster = df_clusters.groupby('Cluster')['AnnualSalary'].mean()
            fig = px.bar(
                x=salary_by_cluster.index.astype(str),
                y=salary_by_cluster.values,
                color=salary_by_cluster.values,
                labels={'x': 'Cluster', 'y': 'Salary ($)'},
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                template="plotly_white",
                font=dict(size=11, color=COLORS['text']),
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")

# ==================== PAGE: PROMOTION GAP ====================
def page_promotion_gap(df):
    """Promotion gap and risk analysis"""
    st.markdown("# ⚠️ Promotion Gap Monitor")
    st.markdown("Identify employees with promotion delays")
    
    try:
        analyzer = CareerAnalysis(df)
        promo_scores = analyzer.calculate_promotion_gap_score()
        
        high_risk = (promo_scores >= 0.6).sum()
        medium_risk = ((promo_scores >= 0.3) & (promo_scores < 0.6)).sum()
        low_risk = (promo_scores < 0.3).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        create_kpi_card(col1, "High Risk", high_risk, icon="🔴")
        create_kpi_card(col2, "Medium Risk", medium_risk, icon="🟠")
        create_kpi_card(col3, "Low Risk", low_risk, icon="🟢")
        create_kpi_card(col4, "Avg Gap Score", f"{promo_scores.mean():.2f}", metric_format="percent")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1.2, 0.8], gap="large")
        
        with col1:
            st.markdown("### Risk Distribution")
            risk_data = pd.Series({
                'High Risk': high_risk,
                'Medium Risk': medium_risk,
                'Low Risk': low_risk
            })
            fig = px.pie(
                values=risk_data.values,
                names=risk_data.index,
                color_discrete_map={
                    'High Risk': COLORS['danger'],
                    'Medium Risk': COLORS['warning'],
                    'Low Risk': COLORS['success']
                },
                hole=0.3
            )
            fig.update_layout(
                template="plotly_white",
                font=dict(size=11, color=COLORS['text']),
                height=400,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### High Risk Employees")
            high_risk_df = df.nlargest(5, 'YearsSinceLastPromotion')[
                ['EmployeeID', 'FirstName', 'YearsAtCompany', 'YearsSinceLastPromotion']
            ].copy()
            high_risk_df.columns = ['ID', 'Name', 'Tenure', 'Gap']
            st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

# ==================== PAGE: RETENTION ====================
def page_retention(df):
    """Retention opportunity analysis"""
    st.markdown("# 🎯 Retention Opportunities")
    st.markdown("Priority employees for retention focus")
    
    try:
        analyzer = CareerAnalysis(df)
        retention_index = analyzer.calculate_retention_opportunity_index()
        
        critical = (retention_index >= 0.7).sum()
        high = ((retention_index >= 0.5) & (retention_index < 0.7)).sum()
        medium = ((retention_index >= 0.3) & (retention_index < 0.5)).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        create_kpi_card(col1, "Critical", critical, icon="🔴")
        create_kpi_card(col2, "High", high, icon="🟠")
        create_kpi_card(col3, "Medium", medium, icon="🟡")
        create_kpi_card(col4, "Avg Index", f"{retention_index.mean():.2f}")
        
        st.markdown("---")
        
        st.markdown("### Critical Priority Employees")
        critical_df = df[retention_index >= 0.7][
            ['EmployeeID', 'FirstName', 'Department', 'Position', 'AnnualSalary']
        ].head(15).copy()
        st.dataframe(critical_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

# ==================== PAGE: MANAGER INSIGHTS ====================
def page_manager_insights(df):
    """Manager and department insights"""
    st.markdown("# 👔 Manager Insights")
    st.markdown("Department and manager effectiveness analysis")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### Employees by Department")
        dept_counts = df['Department'].value_counts()
        fig = px.bar(
            x=dept_counts.index,
            y=dept_counts.values,
            color=dept_counts.values,
            color_continuous_scale="Blues",
            labels={'x': 'Department', 'y': 'Count'}
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=11, color=COLORS['text']),
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Avg Performance by Department")
        perf_by_dept = df.groupby('Department')['PerformanceRating'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=perf_by_dept.index,
            y=perf_by_dept.values,
            color=perf_by_dept.values,
            color_continuous_scale="Greens",
            labels={'x': 'Department', 'y': 'Rating'}
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(size=11, color=COLORS['text']),
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Department Summary")
    dept_summary = df.groupby('Department').agg({
        'EmployeeID': 'count',
        'AnnualSalary': 'mean',
        'PerformanceRating': 'mean',
        'YearsAtCompany': 'mean'
    }).round(2)
    dept_summary.columns = ['Employees', 'Avg Salary', 'Avg Rating', 'Avg Tenure']
    st.dataframe(dept_summary, use_container_width=True)

# ==================== PAGE: FULL ANALYSIS ====================
def page_full_analysis(df):
    """Comprehensive analysis report"""
    st.markdown("# 📊 Full Analysis Report")
    
    with st.expander("📋 **Dataset Overview**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        create_kpi_card(col1, "Employees", len(df))
        create_kpi_card(col2, "Departments", df['Department'].nunique())
        create_kpi_card(col3, "Avg Tenure", f"{df['YearsAtCompany'].mean():.1f} yrs")
        create_kpi_card(col4, "Avg Salary", df['AnnualSalary'].mean(), metric_format="currency")
    
    with st.expander("👥 **Demographics**"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gender Distribution**")
            gender_dist = df['Gender'].value_counts()
            fig = px.pie(values=gender_dist.values, names=gender_dist.index)
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Age Distribution**")
            fig = px.histogram(df, x='Age', nbins=20, color_discrete_sequence=[COLORS['primary']])
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 **Career Metrics**"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Years at Company", f"{df['YearsAtCompany'].mean():.1f}")
            st.metric("Avg Years in Role", f"{df['YearsInCurrentRole'].mean():.1f}")
        with col2:
            st.metric("Avg Years Since Promotion", f"{df['YearsSinceLastPromotion'].mean():.1f}")
            st.metric("Avg Performance", f"{df['PerformanceRating'].mean():.2f}/5.0")

# ==================== MAIN APPLICATION ====================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📈 Data Overview", "🔍 Career Clusters",
             "⚠️ Promotion Gap", "🎯 Retention", "👔 Manager Insights",
             "📊 Full Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data")
        
        if st.button("📊 Load Sample Data", use_container_width=True):
            with st.spinner("Loading data..."):
                df = load_sample_data()
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
        
        uploaded_file = st.file_uploader("Or upload CSV:", type=['csv'])
        if uploaded_file:
            with st.spinner("Processing upload..."):
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully!")
                st.rerun()
        
        st.markdown("---")
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.caption(f"📌 {len(st.session_state.df)} records loaded")
    
    # Main content
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        if page == "🏠 Home":
            page_home()
        elif page == "📈 Data Overview":
            page_data_overview(df)
        elif page == "🔍 Career Clusters":
            page_career_clusters(df)
        elif page == "⚠️ Promotion Gap":
            page_promotion_gap(df)
        elif page == "🎯 Retention":
            page_retention(df)
        elif page == "👔 Manager Insights":
            page_manager_insights(df)
        elif page == "📊 Full Analysis":
            page_full_analysis(df)
    else:
        # Welcome screen
        st.markdown("# 📊 CareerGapAI")
        st.markdown("### HR Career Progression Analytics Platform")
        st.markdown("---")
        
        col1, col2 = st.columns([1.5, 1], gap="large")
        
        with col1:
            st.markdown("""
            ## Welcome to CareerGapAI
            
            Your modern analytics platform for:
            
            ✓ **Career Analysis** - Understand progression patterns  
            ✓ **Risk Detection** - Identify at-risk employees early  
            ✓ **Data-Driven Insights** - Strategic HR decisions  
            ✓ **Retention Strategy** - Optimize team stability  
            ✓ **Performance Tracking** - Monitor key metrics  
            
            ---
            
            ### Get Started
            Click "📊 Load Sample Data" in the sidebar to explore with 1200 realistic employee records.
            """)
        
        with col2:
            st.info("""
            ### Features
            • 1200+ Employee Records
            • 19 Data Points
            • 7 Analysis Pages
            • Interactive Charts
            • ML Clustering
            • Risk Scoring
            """)

if __name__ == "__main__":
    main()
