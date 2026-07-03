"""
CareerGapAI - Executive Career Intelligence Dashboard
A premium workforce analytics experience for leadership and talent teams.
"""

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import FeatureEngineer, DataPreprocessor, CareerPathClustering, CareerAnalysis
from src.career_advisor import CareerAdvisor
from src.executive_insights import ExecutiveInsights
from src.retention_risk import RetentionRiskScorer

try:
    from fpdf import FPDF
except Exception:  # pragma: no cover
    FPDF = None


st.set_page_config(
    page_title="CareerGapAI | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg-dark: #0f1720;
        --panel-dark: #11151a;
        --sidebar-bg: #ffffff;
        --ink-light: #f3efe8;
        --brown: #6d4c41;
        --beige: #f7efe8;
        --accent: #a67c52;
        --muted: #bfa58f;
        --success: #5d7a45;
        --danger: #a94442;
    }
    .stApp { background: var(--bg-dark); color: var(--ink-light); }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg); color: #3b2a22; padding: 20px 18px; border-right: 1px solid #efe6dd;
    }
    .css-1d391kg { background: transparent } /* avoid extra white panel behind sidebar */
    .hero { background: linear-gradient(135deg, rgba(109,76,65,0.95) 0%, rgba(166,124,82,0.95) 100%); border-radius: 12px; padding: 18px 20px; color: white; margin-bottom: 14px; box-shadow: 0 8px 20px rgba(0,0,0,0.35); }
    .kpi-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; padding: 14px; margin-bottom: 10px; }
    .kpi-label { color: var(--muted); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { color: var(--beige); font-size: 1.6rem; font-weight: 700; margin-top: 6px; }
    .small-note { color: var(--muted); font-size: 0.88rem; }
    .stButton > button { background: linear-gradient(135deg, #6d4c41 0%, #a67c52 100%); color: white; border: none; border-radius: 8px; padding: 0.45rem 0.9rem; }
    .upload-box { background: #11151a; padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px; }
    .upload-box, .upload-box * { color: #ffffff !important; }
    .sidebar-title { color: #3b2a22; margin-bottom: 0.25rem; font-size: 1.5rem; font-weight: 700; }
    .sidebar-subtitle { color: #7d6a56; margin-top: 0; margin-bottom: 1rem; font-size: 0.92rem; }
    .status-card { background: #f7efe8; color: #3b2a22; padding: 14px 16px; border-radius: 14px; border: 1px solid #e5dacf; margin-top: 16px; }
    .small-note { color: var(--muted); font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _ensure_output_dir() -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    return out_dir


def _generate_default_dataset(file_path: Path) -> None:
    if file_path.exists():
        return

    import pandas as pd
    import numpy as np

    np.random.seed(42)
    first_names = ["Avery","Jordan","Taylor","Morgan","Alex","Riley","Jamie","Casey","Drew","Quinn","Kendall","Harper","Parker","Cameron","Rowan","Skyler","Emerson","Reese","Logan","Sydney"]
    last_names = ["Brooks","Reed","Fisher","Graham","Wells","Hart","Sharp","Cross","Lane","Price","Ross","Cole","Hayes","Dean","Miles","Chase","Baker","Stone","Gray","Fox"]
    roles = ["Software Engineer","Sales Manager","Product Lead","Business Analyst","Data Scientist","Finance Manager","Operations Lead","Customer Success","Marketing Specialist","HR Partner"]
    departments = ["Engineering","Sales","Product","Finance","Operations","Customer Success","Marketing","HR"]

    rows = []
    for idx in range(500):
        department = np.random.choice(departments, p=[0.18,0.16,0.14,0.12,0.12,0.12,0.10,0.06])
        position = np.random.choice([role for role in roles if department in role or np.random.rand() > 0.4])
        age = int(np.clip(np.random.normal(36, 8), 22, 65))
        tenure = round(np.clip(np.random.normal(5.5, 3.2), 0.5, 20), 1)
        years_since_promo = round(np.clip(np.random.normal(2.4, 2.1), 0.0, tenure), 1)
        performance_rating = round(np.clip(np.random.normal(3.5, 0.8), 1.0, 5.0), 1)
        training_hours = int(np.clip(np.random.poisson(2.4), 0, 8))
        remote_work = np.random.choice(["Yes", "No"], p=[0.34, 0.66])
        attrition_risk = np.random.choice(["Low", "Medium", "High"], p=[0.7, 0.2, 0.1])
        salary = int(np.clip(np.random.normal(85000 + (age - 30) * 1200 + (tenure * 1500), 15000), 38000, 220000))
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"

        rows.append({
            "EmployeeID": f"EMP{idx + 1:04d}",
            "Name": name,
            "Department": department,
            "Position": position,
            "Salary": salary,
            "Age": age,
            "Tenure_Years": tenure,
            "YearsSincePromotion": years_since_promo,
            "Performance_Rating": performance_rating,
            "Training_Hours": training_hours,
            "Attrition_Risk": attrition_risk,
            "Remote_Work": remote_work,
        })

    pd.DataFrame(rows).to_csv(file_path, index=False)


def _prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if df.empty:
        raise ValueError("The uploaded file is empty.")

    rename_map = {
        "EmployeeNumber": "EmployeeID",
        "Employee ID": "EmployeeID",
        "employee_id": "EmployeeID",
        "Tenure_Years": "YearsAtCompany",
        "tenure_years": "YearsAtCompany",
        "YearsSincePromotion": "YearsSinceLastPromotion",
        "years_since_promotion": "YearsSinceLastPromotion",
        "Performance_Rating": "PerformanceRating",
        "performance_rating": "PerformanceRating",
        "Training_Hours": "TrainingTimesLastYear",
        "training_hours": "TrainingTimesLastYear",
        "Salary": "MonthlyIncome",
        "salary": "MonthlyIncome",
        "Position": "JobRole",
        "job_role": "JobRole",
        "Role": "JobRole",
        "Attrition_Risk": "Attrition",
        "attrition_risk": "Attrition",
        "Remote_Work": "RemoteWork",
        "remote_work": "RemoteWork",
        "YearsWithCurrentManager": "YearsWithCurrManager",
        "YearsWithCurrManager": "YearsWithCurrManager",
        "YearsInCurrentRole": "YearsInCurrentRole",
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    if "EmployeeID" not in df.columns:
        df["EmployeeID"] = [f"EMP{idx + 1:04d}" for idx in range(len(df))]
    if "Department" not in df.columns and "dept" in df.columns:
        df["Department"] = df["dept"]
    if "JobRole" not in df.columns:
        df["JobRole"] = df.get("Position", "Unknown") if "Position" in df.columns else "Unknown"

    if "MonthlyIncome" not in df.columns:
        df["MonthlyIncome"] = 100000.0 + np.random.uniform(0, 40000, len(df))
    if "YearsAtCompany" not in df.columns:
        df["YearsAtCompany"] = np.clip(np.random.normal(5, 3, len(df)), 0.1, None)
    if "YearsSinceLastPromotion" not in df.columns:
        df["YearsSinceLastPromotion"] = np.clip(np.random.exponential(2.5, len(df)), 0.0, 15.0)
    if "YearsInCurrentRole" not in df.columns:
        df["YearsInCurrentRole"] = np.clip(df["YearsAtCompany"] * 0.65, 0.1, None)
    if "YearsWithCurrManager" not in df.columns:
        df["YearsWithCurrManager"] = np.clip(df["YearsAtCompany"] * 0.45, 0.1, None)
    if "TrainingTimesLastYear" not in df.columns:
        df["TrainingTimesLastYear"] = np.clip(np.random.poisson(2, len(df)), 0, 8)
    if "PerformanceRating" not in df.columns:
        df["PerformanceRating"] = np.clip(np.random.normal(3.2, 0.7, len(df)), 1, 5)
    if "Age" not in df.columns:
        df["Age"] = np.clip(np.random.normal(38, 9, len(df)), 20, 70)
    if "JobSatisfaction" not in df.columns:
        df["JobSatisfaction"] = np.clip(np.round(2.0 + (df["PerformanceRating"] - 3) * 0.35 + np.random.normal(0, 0.4, len(df))), 1, 5)
    if "WorkLifeBalance" not in df.columns:
        df["WorkLifeBalance"] = np.clip(np.round(2.8 + np.random.normal(0, 0.5, len(df))), 1, 5)

    if "Attrition" in df.columns:
        if df["Attrition"].dtype == object or str(df["Attrition"].dtype) == "category":
            mapping = {"No": 0, "no": 0, "0": 0, "False": 0, "Low": 0, "Medium": 1, "High": 1, "Yes": 1, "yes": 1, "1": 1}
            df["Attrition"] = df["Attrition"].map(mapping).fillna(0).astype(int)
        else:
            df["Attrition"] = df["Attrition"].astype(float).fillna(0)
    else:
        df["Attrition"] = (df["PerformanceRating"] < 2.8).astype(int)

    df["EmployeeID"] = df["EmployeeID"].astype(str)
    df["Department"] = df["Department"].fillna("General").astype(str)
    df["JobRole"] = df["JobRole"].fillna("Professional").astype(str)
    return df


def _build_analysis_frame(uploaded_file=None) -> Tuple[pd.DataFrame, Dict]:
    if uploaded_file is None:
        raise ValueError("Please upload a dataset to begin the analysis.")

    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df = _prepare_dataframe(df)

    engineer = FeatureEngineer(df)
    engineer.create_all_career_metrics()
    df = engineer.add_engineered_features_to_original()

    preprocessor = DataPreprocessor(df)
    prepared = preprocessor.apply_full_preprocessing_pipeline(
        remove_outliers=True,
        handle_missing=True,
        normalize=False,
        encode_categorical=False,
        remove_dups=True,
    )

    model_df = prepared.select_dtypes(include=[np.number]).copy()
    model_df = model_df.drop(columns=[col for col in ["Attrition"] if col in model_df.columns], errors="ignore")
    model_df = model_df.fillna(0)

    clustering = CareerPathClustering(model_df)
    try:
        cluster_results = clustering.determine_optimal_clusters(k_range=range(2, 8))
        clustering.fit_kmeans(n_clusters=cluster_results.get("optimal_k", 3))
        labels = clustering.get_cluster_assignments()
    except Exception:
        # Fallback to a lightweight clustering when the full scan fails
        try:
            clustering.fit_kmeans(n_clusters=3)
            labels = clustering.get_cluster_assignments()
            cluster_results = {"optimal_k": 3}
        except Exception:
            labels = np.zeros(len(model_df), dtype=int)
            cluster_results = {"optimal_k": 1}

    analysis_df = prepared.copy()
    analysis_df["Cluster"] = labels
    analyzer = CareerAnalysis(analysis_df, labels)
    analyzer.calculate_promotion_gap_score()
    analyzer.calculate_stagnation_risk_score()
    analyzer.calculate_retention_opportunity_index()

    analysis_df["PromotionGapRiskScore"] = analyzer.analysis_results["promotion_gap_score"]["values"]
    analysis_df["StagnationRiskScore"] = analyzer.analysis_results["stagnation_risk_score"]["values"]
    analysis_df["RetentionOpportunityScore"] = analyzer.analysis_results["retention_opportunity_index"]["values"]

    risk_df = RetentionRiskScorer.add_risk_scores(analysis_df)
    risk_df = CareerAdvisor.generate_all_recommendations(risk_df, risk_df["RiskScore"])
    risk_df["RiskBand"] = pd.cut(risk_df["RiskScore"], bins=[-1, 35, 65, 101], labels=["Low", "Medium", "High"])
    risk_df["RiskBand"] = risk_df["RiskBand"].fillna("Low")

    metrics = {
        "cluster_count": int(cluster_results["optimal_k"]),
        "risk_summary": RetentionRiskScorer.get_risk_summary(risk_df),
    }
    return risk_df, metrics


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("<div style='margin-bottom:12px; color:#3b2a22; font-size:0.95rem; font-weight:700;'>Filters</div>", unsafe_allow_html=True)
    departments = st.sidebar.multiselect(
        "Department",
        sorted(df["Department"].dropna().unique()),
        default=sorted(df["Department"].dropna().unique()),
        key="department_filter",
    )
    risk_band = st.sidebar.multiselect(
        "Risk Band",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
        key="risk_band_filter",
    )
    min_tenure = st.sidebar.slider("Min tenure (years)", 0.0, 30.0, 0.0, 0.5, key="tenure_filter")
    min_score = st.sidebar.slider("Minimum risk score", 0, 100, 0, 1, key="risk_score_filter")
    st.sidebar.markdown(
        "<div class='small-note'>Use these filters to narrow the dashboard by department, retention risk, tenure, and risk score. The chart and table content will update to show only the matching employee records.</div>",
        unsafe_allow_html=True,
    )

    filtered = df.copy()
    if departments:
        filtered = filtered[filtered["Department"].isin(departments)]
    if risk_band:
        filtered = filtered[filtered["RiskBand"].isin(risk_band)]
    filtered = filtered[filtered.get("YearsAtCompany", pd.Series(0, index=filtered.index)).fillna(0) >= min_tenure]
    filtered = filtered[filtered.get("RiskScore", pd.Series(0, index=filtered.index)).fillna(0) >= min_score]
    return filtered.reset_index(drop=True)


def _render_kpi_card(label: str, value: str, subtext: str, accent: str = "#6d4c41") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{accent};">{value}</div>
            <div class="small-note">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(df: pd.DataFrame, metrics: dict) -> None:
    st.markdown("<div class='hero'><h1>Executive Career Intelligence</h1><p>Premium workforce insights for retention, promotion, and talent planning.</p></div>", unsafe_allow_html=True)
    risk_summary = metrics["risk_summary"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_kpi_card("Employees", f"{int(risk_summary['total_employees'])}", "Active workforce in view", "#6d4c41")
    with col2:
        _render_kpi_card("Avg risk", f"{float(risk_summary['avg_risk_score']):.1f}", "Retention risk score", "#a67c52")
    with col3:
        _render_kpi_card("High risk", f"{float(risk_summary['high_risk_pct']):.1f}%", "Employees requiring action", "#a94442")
    with col4:
        _render_kpi_card("Avg tenure", f"{float(df['YearsAtCompany'].mean()):.1f} yrs", "Career tenure", "#5d7a45")

    left, right = st.columns([1.2, 1.0])
    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(risk_summary["avg_risk_score"]),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Average Retention Risk"},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "#6d4c41"},
                    "steps": [
                        {"range": [0, 35], "color": "#e8f4ea"},
                        {"range": [35, 65], "color": "#f9ecd1"},
                        {"range": [65, 100], "color": "#f5d7d7"},
                    ],
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dept_risk = df.groupby(["Department", "RiskBand"]).size().reset_index(name="Employees")
        fig = px.bar(dept_risk, x="Department", y="Employees", color="RiskBand", barmode="group", color_discrete_map={"Low": "#5d7a45", "Medium": "#c97f26", "High": "#a94442"}, title="Risk Distribution by Department")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sunburst_df = df[["Department", "JobRole", "RiskBand"]].copy()
        fig = px.sunburst(sunburst_df, path=["Department", "JobRole", "RiskBand"], title="Career Risk Sunburst")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_risk = df.sort_values("RiskScore", ascending=False).head(10)[["EmployeeID", "Department", "JobRole", "RiskScore", "RiskBand"]]
        st.dataframe(top_risk, use_container_width=True, hide_index=True)


def _render_explorer(df: pd.DataFrame) -> None:
    st.markdown("<div class='hero'><h2>Employee Explorer</h2><p>Inspect employee profiles and career progression patterns with premium filters.</p></div>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No employees match the active filters.")
        return

    scatter = px.scatter(df, x="YearsSinceLastPromotion", y="RiskScore", color="Department", size="MonthlyIncome", hover_data=["EmployeeID", "JobRole"], title="Risk vs Promotion Gap")
    st.plotly_chart(scatter, use_container_width=True)

    selected_employee = st.selectbox("Select employee", df["EmployeeID"].tolist())
    employee = df[df["EmployeeID"] == selected_employee].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_kpi_card("Employee", employee["EmployeeID"], employee["JobRole"], "#6d4c41")
    with col2:
        _render_kpi_card("Risk", f"{employee['RiskScore']:.1f}", employee["RiskBand"], "#a94442")
    with col3:
        _render_kpi_card("Promotion gap", f"{employee['YearsSinceLastPromotion']:.1f} yrs", "Since last promotion", "#a67c52")
    with col4:
        _render_kpi_card("Tenure", f"{employee['YearsAtCompany']:.1f} yrs", "At company", "#5d7a45")

    st.markdown("### Employee Snapshot")
    profile = pd.DataFrame(
        {
            "Metric": ["Department", "Position", "Performance", "Training", "Manager tenure", "Satisfaction", "Work-life balance"],
            "Value": [
                employee["Department"],
                employee["JobRole"],
                f"{employee['PerformanceRating']:.1f}/5",
                f"{employee['TrainingTimesLastYear']:.0f} sessions",
                f"{employee['YearsWithCurrManager']:.1f} yrs",
                f"{employee['JobSatisfaction']:.1f}/5",
                f"{employee['WorkLifeBalance']:.1f}/5",
            ],
        }
    )
    st.dataframe(profile, hide_index=True, use_container_width=True)


def _render_risk_intelligence(df: pd.DataFrame) -> None:
    st.markdown("<div class='hero'><h2>Risk Intelligence</h2><p>Explainable AI view of retention risk and intervention priority.</p></div>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No employees match the active filters.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_kpi_card("High risk", f"{int((df['RiskBand'] == 'High').sum())}", "Employees needing intervention", "#a94442")
    with col2:
        _render_kpi_card("Medium risk", f"{int((df['RiskBand'] == 'Medium').sum())}", "Monitor closely", "#c97f26")
    with col3:
        _render_kpi_card("Low risk", f"{int((df['RiskBand'] == 'Low').sum())}", "Stable cohort", "#5d7a45")

    left, right = st.columns(2)
    with left:
        heatmap = pd.crosstab(df["Department"], df["RiskBand"])
        fig = px.imshow(heatmap, text_auto=True, color_continuous_scale="Peach")
        fig.update_layout(title="Department × Risk Band")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.treemap(df, path=["Department", "JobRole"], values="RiskScore", title="Risk by Department and Role")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Explainability Snapshot")
    sample = df.sort_values("RiskScore", ascending=False).head(5)
    for _, row in sample.iterrows():
        reasons = []
        if row["YearsSinceLastPromotion"] >= 3:
            reasons.append("Promotion gap is elevated")
        if row["YearsInCurrentRole"] >= 4:
            reasons.append("Role tenure is extended")
        if row["TrainingTimesLastYear"] <= 1:
            reasons.append("Training participation is low")
        if row["JobSatisfaction"] <= 2.5:
            reasons.append("Satisfaction is below target")
        if row["YearsWithCurrManager"] <= 1.5:
            reasons.append("Manager continuity is low")
        if not reasons:
            reasons.append("Balanced growth signals")
        with st.expander(f"{row['EmployeeID']} — {row['Department']} / {row['JobRole']}"):
            st.write(f"Risk score: {row['RiskScore']:.1f}")
            st.write("Why this score: " + "; ".join(reasons))


def _render_advisor(df: pd.DataFrame) -> None:
    st.markdown("<div class='hero'><h2>AI Career Advisor</h2><p>Personalized interventions for growth, retention, and leadership readiness.</p></div>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No employees match the active filters.")
        return

    employee_id = st.selectbox("Select an employee", df["EmployeeID"].tolist())
    employee = df[df["EmployeeID"] == employee_id].iloc[0]
    risk_score = float(employee["RiskScore"])
    recommendations = CareerAdvisor.generate_recommendations(employee, risk_score)

    col1, col2 = st.columns(2)
    with col1:
        _render_kpi_card("Priority", CareerAdvisor.get_priority_level(risk_score), "Recommended action timing", "#a94442")
    with col2:
        _render_kpi_card("Archetype", CareerAdvisor.get_employee_archetype(employee, risk_score), "Talent profile", "#6d4c41")

    st.markdown("### Suggested Actions")
    for rec in recommendations:
        st.info(f"{rec['text']} — {rec['reason']} ({rec['priority']})")

    st.markdown("### Next Steps")
    for step in CareerAdvisor.get_next_steps(employee, risk_score, recommendations):
        st.write(step)

    st.markdown("### Executive Insights")
    for insight in ExecutiveInsights.generate_all_insights(df)[:4]:
        st.success(f"{insight['title']}: {insight['description']}")


def _create_pdf_report(df: pd.DataFrame) -> bytes:
    if FPDF is None:
        raise RuntimeError("The FPDF package is not installed.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CareerGapAI Executive Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary = [
        f"Employees in view: {len(df)}",
        f"Average risk score: {df['RiskScore'].mean():.1f}",
        f"High risk employees: {(df['RiskBand'] == 'High').sum()}",
        f"Average promotion gap: {df['YearsSinceLastPromotion'].mean():.1f} years",
    ]
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    for line in summary:
        pdf.cell(0, 6, line, ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Top risks", ln=True)
    pdf.set_font("Arial", "", 9)
    top_rows = df.sort_values("RiskScore", ascending=False).head(8)[["EmployeeID", "Department", "JobRole", "RiskScore", "RiskBand"]]
    for _, row in top_rows.iterrows():
        pdf.cell(0, 5, f"{row['EmployeeID']} | {row['Department']} | {row['JobRole']} | Risk {row['RiskScore']:.1f} | {row['RiskBand']}", ln=True)

    return pdf.output(dest="S").encode("latin-1")


def _render_report(df: pd.DataFrame) -> None:
    st.markdown("<div class='hero'><h2>Executive Report</h2><p>Download a polished PDF summary for leadership review.</p></div>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No employees match the active filters.")
        return

    if st.button("Export PDF report"):
        try:
            pdf_bytes = _create_pdf_report(df)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="careergapai_executive_report.pdf",
                mime="application/pdf",
            )
            st.success("Executive PDF prepared successfully.")
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")

    st.markdown("### Report Snapshot")
    st.dataframe(df[["EmployeeID", "Department", "JobRole", "RiskScore", "RiskBand", "YearsSinceLastPromotion"]].sort_values("RiskScore", ascending=False).head(20), use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.markdown("<div class='sidebar-title'>CareerGapAI</div><div class='sidebar-subtitle'>Premium career intelligence</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='upload-box'><strong>Drag and drop file here</strong><br><span style='color:#ccc; font-size:0.9rem;'>Limit 200MB per file · CSV only</span></div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload employee CSV", type=["csv"], help="Upload a dataset file with employee records.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='margin-bottom:10px; color:#3b2a22; font-size:0.95rem; font-weight:700;'>Navigation</div>", unsafe_allow_html=True)
    nav = st.sidebar.radio("", ["Executive Overview", "Employee Explorer", "Risk Intelligence", "AI Career Advisor", "Executive Report"], index=0, key='nav')
    st.sidebar.markdown("---")

    if uploaded_file is None:
        st.sidebar.markdown("<div class='status-card'>Awaiting uploaded employee dataset to begin analysis.</div>", unsafe_allow_html=True)
        st.info("Please upload a CSV file to start the dashboard.")
        return

    st.sidebar.markdown("<div class='status-card' style='background:#ddecf3; color:#1e3a50;'>Custom dataset uploaded and ready to analyze.</div>", unsafe_allow_html=True)

    try:
        analysis_df, metrics = _build_analysis_frame(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to load the uploaded file: {exc}")
        st.info("Please upload a valid CSV with employee records.")
        return

    st.sidebar.markdown(f"<div class='status-card'>Loaded {len(analysis_df)} employees from active dataset</div>", unsafe_allow_html=True)
    filtered_df = _apply_filters(analysis_df)

    if nav == "Executive Overview":
        _render_overview(filtered_df, metrics)
    elif nav == "Employee Explorer":
        _render_explorer(filtered_df)
    elif nav == "Risk Intelligence":
        _render_risk_intelligence(filtered_df)
    elif nav == "AI Career Advisor":
        _render_advisor(filtered_df)
    else:
        _render_report(filtered_df)


if __name__ == "__main__":
    main()
