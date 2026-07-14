"""
CareerGapAI - Clean Minimal Dashboard
Light theme, no clutter, professional design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CareerGapAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== LIGHT THEME COLORS ====================
st.markdown("""
<style>
    /* Light theme */
    .main { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #F7F9FC; border-right: 1px solid #E5E7EB; }
    
    /* Headers */
    h1 { color: #111827 !important; font-weight: 700 !important; font-size: 2rem !important; }
    h2 { color: #1F2937 !important; font-weight: 600 !important; font-size: 1.4rem !important; border-bottom: 2px solid #3B82F6 !important; padding-bottom: 0.5rem !important; }
    h3 { color: #374151 !important; font-weight: 600 !important; }
    
    /* Text */
    p, span, label { color: #4B5563 !important; }
    
    /* Buttons */
    .stButton > button { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important; color: white !important; border: none !important; border-radius: 6px !important; padding: 0.6rem 1.5rem !important; font-weight: 600 !important; }
    .stButton > button:hover { box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important; }
    
    /* Input fields */
    .stTextInput > div > div > input, .stSelectbox > div > div > select { background: #F3F4F6 !important; border: 1px solid #D1D5DB !important; color: #111827 !important; border-radius: 6px !important; }
    
    /* Metric cards */
    [data-testid="stMetricContainer"] { background: #F8FAFC !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; padding: 1.2rem !important; }
    
    /* Tables */
    .stDataFrame { background: #FFFFFF !important; }
    
    /* Expander */
    [data-testid="stExpander"] { background: #F8FAFC !important; border: 1px solid #E2E8F0 !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# ==================== LOAD DATA ====================
@st.cache_resource
def load_data():
    """Load employee dataset"""
    try:
        df = pd.read_csv('data/employees.csv')
        return df
    except:
        return None

# ==================== PAGES ====================

def page_home():
    """Home page - Simple overview"""
    st.markdown("# 📊 CareerGapAI")
    st.markdown("Employee Career & Retention Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(st.session_state.df))
    with col2:
        st.metric("Departments", st.session_state.df['Department'].nunique())
    with col3:
        st.metric("Avg Salary", f"${st.session_state.df['Salary'].mean():,.0f}")
    with col4:
        st.metric("Avg Rating", f"{st.session_state.df['Performance_Rating'].mean():.1f}/5")

def page_employees():
    """Employee database view"""
    st.markdown("# 👥 Employee Data")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_filter = st.multiselect("Department", st.session_state.df['Department'].unique(), default=st.session_state.df['Department'].unique())
    with col2:
        risk_filter = st.multiselect("Attrition Risk", st.session_state.df['Attrition_Risk'].unique(), default=st.session_state.df['Attrition_Risk'].unique())
    with col3:
        search = st.text_input("Search by Name")
    
    # Filter data
    df_filtered = st.session_state.df[
        (st.session_state.df['Department'].isin(dept_filter)) & 
        (st.session_state.df['Attrition_Risk'].isin(risk_filter))
    ]
    
    if search:
        df_filtered = df_filtered[df_filtered['Name'].str.contains(search, case=False)]
    
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(df_filtered)} employees")

def page_analytics():
    """Analytics and insights"""
    st.markdown("# 📈 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Employees by Department")
        dept_data = st.session_state.df['Department'].value_counts()
        fig = px.bar(x=dept_data.index, y=dept_data.values, color_discrete_sequence=['#3B82F6'])
        fig.update_layout(template="plotly_white", height=300, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Attrition Risk Distribution")
        risk_data = st.session_state.df['Attrition_Risk'].value_counts()
        fig = px.pie(values=risk_data.values, names=risk_data.index, color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'])
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Salary Distribution")
        fig = px.box(st.session_state.df, y='Salary', color_discrete_sequence=['#2563EB'])
        fig.update_layout(template="plotly_white", height=300, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Performance by Department")
        perf_data = st.session_state.df.groupby('Department')['Performance_Rating'].mean().sort_values(ascending=False)
        fig = px.bar(x=perf_data.index, y=perf_data.values, color_discrete_sequence=['#06B6D4'])
        fig.update_layout(template="plotly_white", height=300, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

def page_high_risk():
    """High risk employees"""
    st.markdown("# ⚠️ High Risk Employees")
    
    high_risk = st.session_state.df[st.session_state.df['Attrition_Risk'] == 'High'].sort_values('Tenure_Years')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Risk Count", len(high_risk))
    with col2:
        st.metric("% of Total", f"{len(high_risk)/len(st.session_state.df)*100:.1f}%")
    with col3:
        st.metric("Avg Tenure", f"{high_risk['Tenure_Years'].mean():.1f} years")
    
    st.markdown("### List")
    st.dataframe(high_risk[['EmployeeID', 'Name', 'Department', 'Salary', 'Tenure_Years', 'Performance_Rating']], use_container_width=True, hide_index=True)

def page_statistics():
    """Summary statistics"""
    st.markdown("# 📊 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Salary Stats")
        st.metric("Average", f"${st.session_state.df['Salary'].mean():,.0f}")
        st.metric("Min - Max", f"${st.session_state.df['Salary'].min():,.0f} - ${st.session_state.df['Salary'].max():,.0f}")
        st.metric("Median", f"${st.session_state.df['Salary'].median():,.0f}")
    
    with col2:
        st.markdown("### Tenure Stats")
        st.metric("Average", f"{st.session_state.df['Tenure_Years'].mean():.1f} years")
        st.metric("Min - Max", f"{st.session_state.df['Tenure_Years'].min():.1f} - {st.session_state.df['Tenure_Years'].max():.1f} years")
        st.metric("Median", f"{st.session_state.df['Tenure_Years'].median():.1f} years")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Stats")
        st.metric("Average Rating", f"{st.session_state.df['Performance_Rating'].mean():.2f}/5")
        st.metric("High Performers (>4.0)", (st.session_state.df['Performance_Rating'] > 4.0).sum())
    
    with col2:
        st.markdown("### Work Stats")
        remote = (st.session_state.df['Remote_Work'] == 'Yes').sum()
        st.metric("Remote Workers", f"{remote} ({remote/len(st.session_state.df)*100:.1f}%)")
        st.metric("Avg Training Hours", f"{st.session_state.df['Training_Hours'].mean():.0f} hrs")

# ==================== MAIN ====================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Menu")
        page = st.radio("Select:", ["Home", "Employees", "Analytics", "High Risk", "Statistics"], label_visibility="collapsed")
        
        st.markdown("---")
        
        if st.button("📥 Load Data", use_container_width=True):
            with st.spinner("Loading..."):
                df = load_data()
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success("✅ Loaded!")
                    st.rerun()
                else:
                    st.error("❌ File not found")
        
        st.markdown("---")
        
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.caption(f"📌 {len(st.session_state.df)} records loaded")
            st.caption("Ready for analysis")
    
    # Main content
    if st.session_state.data_loaded and st.session_state.df is not None:
        if page == "Home":
            page_home()
        elif page == "Employees":
            page_employees()
        elif page == "Analytics":
            page_analytics()
        elif page == "High Risk":
            page_high_risk()
        elif page == "Statistics":
            page_statistics()
    else:
        st.markdown("# 📊 CareerGapAI")
        st.info("Click **📥 Load Data** in the sidebar to begin", icon="ℹ️")

if __name__ == "__main__":
    main()
