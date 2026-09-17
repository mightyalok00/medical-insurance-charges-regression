"""Professional Streamlit dashboard for the Medical Insurance Charges Regression project."""

from pathlib import Path
from typing import Dict
import math

import altair as alt
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "insurance_cleaned.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "Linear Regression": MODELS_DIR / "linear_regression.pkl",
    "Polynomial Regression": MODELS_DIR / "polynomial_regression.pkl",
    "Decision Tree Regression": MODELS_DIR / "decision_tree_regression.pkl",
}

COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
DIAGNOSTICS_PATH = REPORTS_DIR / "model_diagnostics.csv"
CV_PATH = REPORTS_DIR / "cross_validation_results.csv"
SEGMENTS_PATH = REPORTS_DIR / "predicted_cost_segments.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "decision_tree_feature_importance.csv"
REPORT_MD_PATH = REPORTS_DIR / "project_report.md"
REPORT_PDF_PATH = REPORTS_DIR / "project_report.pdf"

st.set_page_config(
    page_title="Medical Insurance ML Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 1.4rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: rgba(120,120,120,.06); border: 1px solid rgba(120,120,120,.18); border-radius: 14px; padding: .85rem;}
      .hero {padding: 1.3rem 1.45rem; border: 1px solid rgba(120,120,120,.20); border-radius: 18px; background: linear-gradient(135deg, rgba(37,99,235,.10), rgba(16,185,129,.07)); margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .82;}
      .best-model {padding: .95rem 1rem; border: 1px solid rgba(16,185,129,.30); background: rgba(16,185,129,.07); border-radius: 12px; margin: .6rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return clean_columns(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def normalize_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = clean_columns(df)
    if "Unnamed: 0" in out.columns:
        out = out.rename(columns={"Unnamed: 0": "Feature"})
    for alias in ["feature", "features", "Feature Name", "Feature_Name"]:
        if "Feature" not in out.columns and alias in out.columns:
            out = out.rename(columns={alias: "Feature"})
    for alias in ["Importance", "feature_importance", "Feature Importance", "Feature_Importance"]:
        if "importance" not in out.columns and alias in out.columns:
            out = out.rename(columns={alias: "importance"})
    unnamed = [c for c in out.columns if str(c).lower().startswith("unnamed:")]
    return out.drop(columns=unnamed, errors="ignore")


@st.cache_resource
def load_models() -> tuple[Dict[str, object], Dict[str, str]]:
    models: Dict[str, object] = {}
    errors: Dict[str, str] = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            errors[name] = f"Missing file: {path.name}"
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as exc:
            errors[name] = str(exc)
    return models, errors


@st.cache_data
def load_assets():
    if not DATA_PATH.exists():
        st.error(f"Required dataset not found: {DATA_PATH}")
        st.stop()

    data = clean_columns(pd.read_csv(DATA_PATH))
    comparison = read_csv_if_exists(COMPARISON_PATH)
    diagnostics = read_csv_if_exists(DIAGNOSTICS_PATH)
    cv_results = read_csv_if_exists(CV_PATH)
    segments = read_csv_if_exists(SEGMENTS_PATH)
    importance = normalize_feature_importance(read_csv_if_exists(FEATURE_IMPORTANCE_PATH))
    return data, comparison, diagnostics, cv_results, segments, importance


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def get_best_model_name(df: pd.DataFrame) -> str:
    if not df.empty and {"Model", "Test RMSE"}.issubset(df.columns):
        rmse = pd.to_numeric(df["Test RMSE"], errors="coerce")
        if rmse.notna().any():
            return str(df.loc[rmse.idxmin(), "Model"])
    return "Decision Tree Regression"


def model_metric_chart(df: pd.DataFrame, metric: str):
    p = df[["Model", metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna()
    return (
        alt.Chart(p)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X(field="Model", type="nominal", title=None, sort=None),
            y=alt.Y(field=metric, type="quantitative", title=metric),
            tooltip=[
                alt.Tooltip(field="Model", type="nominal", title="Model"),
                alt.Tooltip(field=metric, type="quantitative", title=metric, format=",.3f"),
            ],
        )
        .properties(height=350)
    )


def feature_importance_chart(df: pd.DataFrame):
    p = df[["Feature", "importance"]].copy()
    p["importance"] = pd.to_numeric(p["importance"], errors="coerce")
    p = p.dropna().sort_values("importance", ascending=False).head(12)
    return (
        alt.Chart(p)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(field="importance", type="quantitative", title="Feature importance"),
            y=alt.Y(field="Feature", type="nominal", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip(field="Feature", type="nominal"),
                alt.Tooltip(field="importance", type="quantitative", format=".4f"),
            ],
        )
        .properties(height=390)
    )


def reset_filters():
    for key in ["age_filter", "bmi_filter", "children_filter", "sex_filter", "smoker_filter", "region_filter"]:
        st.session_state.pop(key, None)


models, model_errors = load_models()
data, comparison, diagnostics, cv_results, segments, importance = load_assets()

required = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
missing = required - set(data.columns)
if missing:
    st.error("Cleaned dataset is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

best_model_name = get_best_model_name(comparison)
available_models = [name for name in MODEL_PATHS if name in models]

age_min, age_max = int(data.age.min()), int(data.age.max())
bmi_min, bmi_max = float(data.bmi.min()), float(data.bmi.max())
# Expand BMI slider bounds outward to the nearest 0.1 so default filters include every row.
bmi_slider_min = math.floor(bmi_min * 10) / 10
bmi_slider_max = math.ceil(bmi_max * 10) / 10
children_min, children_max = int(data.children.min()), int(data.children.max())
sex_options = sorted(data.sex.dropna().astype(str).unique().tolist())
smoker_options = sorted(data.smoker.dropna().astype(str).unique().tolist())
region_options = sorted(data.region.dropna().astype(str).unique().tolist())
q1_charge = float(data.charges.quantile(.25))
median_charge = float(data.charges.median())
q3_charge = float(data.charges.quantile(.75))

st.sidebar.markdown("## 🏥 Project Navigator")
st.sidebar.caption("Medical Insurance Charges Regression")

st.sidebar.markdown("### 🧠 Prediction model")
selector_options = ["Compare all models"] + available_models
selected_model = st.sidebar.selectbox(
    "Select model",
    options=selector_options,
    index=0,
    key="global_model_selector",
    help="This selector controls the Prediction Lab.",
)

if available_models:
    st.sidebar.success(f"Loaded {len(available_models)}/3 model files")
else:
    st.sidebar.error("No trained models could be loaded")

if model_errors:
    with st.sidebar.expander("Model load diagnostics"):
        for name, error in model_errors.items():
            st.write(f"**{name}:** {error}")

st.sidebar.markdown("### 🎛️ Dataset filters")
age_filter = st.sidebar.slider("🎂 Age", age_min, age_max, (age_min, age_max), key="age_filter")
bmi_filter = st.sidebar.slider(
    "⚖️ BMI",
    min_value=bmi_slider_min,
    max_value=bmi_slider_max,
    value=(bmi_slider_min, bmi_slider_max),
    step=.1,
    key="bmi_filter",
)
children_filter = st.sidebar.slider(
    "👶 Children", children_min, children_max, (children_min, children_max), key="children_filter"
)
sex_filter = st.sidebar.multiselect("🚻 Sex", sex_options, default=sex_options, key="sex_filter")
smoker_filter = st.sidebar.multiselect("🚬 Smoker", smoker_options, default=smoker_options, key="smoker_filter")
region_filter = st.sidebar.multiselect("📍 Region", region_options, default=region_options, key="region_filter")

filtered_data = data[
    data.age.between(*age_filter)
    & data.bmi.between(*bmi_filter)
    & data.children.between(*children_filter)
    & data.sex.astype(str).isin(sex_filter)
    & data.smoker.astype(str).isin(smoker_filter)
    & data.region.astype(str).isin(region_filter)
].copy()

st.sidebar.metric("Filtered records", f"{len(filtered_data):,}")
st.sidebar.caption(f"Best saved holdout model: **{best_model_name}**")
st.sidebar.button("↺ Reset dataset filters", on_click=reset_filters, width="stretch")

st.markdown(
    """
    <div class="hero">
      <h1>🏥 Medical Insurance Charges Regression</h1>
      <p>Professional ML portfolio dashboard comparing Linear Regression, Polynomial Regression, and Decision Tree Regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

head = st.columns(5)
head[0].metric("Dataset rows", f"{len(data):,}")
head[1].metric("Models loaded", f"{len(available_models)}/3")
head[2].metric("Best test model", best_model_name.replace(" Regression", ""))
head[3].metric("Median charge", format_currency(median_charge))
head[4].metric("Filtered rows", f"{len(filtered_data):,}")

if len(filtered_data) == len(data):
    st.caption("All dataset rows are currently included in the sidebar filters.")
else:
    st.caption(f"{len(data) - len(filtered_data):,} row(s) are currently excluded by the sidebar filters.")

st.caption("Educational portfolio project only — not for real insurance pricing, underwriting, eligibility, or adverse-action decisions.")

overview_tab, prediction_tab, performance_tab, insights_tab, explorer_tab, files_tab = st.tabs([
    "🏠 Executive Overview", "🔮 Prediction Lab", "📊 Model Performance",
    "🧠 Drivers & Insights", "🔎 Data Explorer", "📄 Project Files"
])

with overview_tab:
    st.subheader("Executive overview")
    if filtered_data.empty:
        st.warning("No rows match the current filters.")
    else:
        c = st.columns(4)
        c[0].metric("Average charge", format_currency(float(filtered_data.charges.mean())))
        c[1].metric("Average age", f"{filtered_data.age.mean():.1f}")
        c[2].metric("Average BMI", f"{filtered_data.bmi.mean():.1f}")
        smoker_share = filtered_data.smoker.astype(str).str.lower().eq("yes").mean() * 100
        c[3].metric("Smoker share", f"{smoker_share:.1f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Charge distribution")
            hist = alt.Chart(filtered_data).mark_bar().encode(
                x=alt.X("charges:Q", bin=alt.Bin(maxbins=35), title="Charges"),
                y=alt.Y("count():Q", title="Records")
            ).properties(height=320)
            st.altair_chart(hist, width="stretch")
        with right:
            st.markdown("#### Charges by smoking status")
            box = alt.Chart(filtered_data).mark_boxplot(size=45).encode(
                x=alt.X("smoker:N", title="Smoker"),
                y=alt.Y("charges:Q", title="Charges")
            ).properties(height=320)
            st.altair_chart(box, width="stretch")

    st.markdown(
        f'<div class="best-model"><strong>Best saved holdout model:</strong> {best_model_name}. Selection is based on the lowest Test RMSE in <code>reports/model_comparison.csv</code>.</div>',
        unsafe_allow_html=True,
    )

with prediction_tab:
    st.subheader("Prediction lab")
    st.info(f"Current model selection: **{selected_model}**. Change it anytime from the left sidebar.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", age_min, age_max, int(data.age.median()), key="pred_age")
        sex = st.selectbox("Sex", sex_options, key="pred_sex")
    with c2:
        bmi = st.slider(
            "BMI",
            min_value=bmi_slider_min,
            max_value=bmi_slider_max,
            value=round(float(data.bmi.median()), 1),
            step=.1,
            key="pred_bmi",
        )
        children = st.slider("Children", children_min, children_max, int(data.children.median()), key="pred_children")
    with c3:
        smoker = st.selectbox("Smoker", smoker_options, key="pred_smoker")
        region = st.selectbox("Region", region_options, key="pred_region")

    profile = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region,
    }])

    if st.button("💰 Generate prediction", type="primary", width="stretch"):
        if not available_models:
            st.error("No trained model artifacts could be loaded. See Model load diagnostics in the sidebar.")
        else:
            try:
                if selected_model == "Compare all models":
                    rows = []
                    for name in available_models:
                        pred = float(models[name].predict(profile)[0])
                        rows.append({"Model": name, "Predicted Charge": pred})
                    pred_df = pd.DataFrame(rows)
                    pred_df["Predicted Charge"] = pred_df["Predicted Charge"].round(2)
                    st.dataframe(pred_df.style.format({"Predicted Charge": "${:,.2f}"}), width="stretch", hide_index=True)
                    chart = alt.Chart(pred_df).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7).encode(
                        x=alt.X("Model:N", title=None, sort=None),
                        y=alt.Y("Predicted Charge:Q", title="Predicted charge"),
                        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Predicted Charge:Q", format="$,.2f")]
                    ).properties(height=330)
                    st.altair_chart(chart, width="stretch")
                    if best_model_name in pred_df.Model.values:
                        best_pred = float(pred_df.loc[pred_df.Model == best_model_name, "Predicted Charge"].iloc[0])
                        st.success(f"Portfolio default — **{best_model_name}**: {format_currency(best_pred)}")
                else:
                    pred = float(models[selected_model].predict(profile)[0])
                    st.metric(f"Predicted charge — {selected_model}", format_currency(pred))
                    if pred < q1_charge:
                        band = "Lower-cost band"
                    elif pred > q3_charge:
                        band = "Higher-cost band"
                    else:
                        band = "Mid-cost band"
                    st.info(f"Descriptive cost segment: **{band}**. This is not an underwriting classification.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with performance_tab:
    st.subheader("Model performance")
    if comparison.empty:
        st.warning("Model comparison report is unavailable.")
    else:
        st.dataframe(comparison, width="stretch", hide_index=True)
        metrics = [c for c in ["Test RMSE", "Test MAE", "Test MSE", "Test R2", "Test R²"] if c in comparison.columns]
        if "Model" in comparison.columns and metrics:
            metric = st.selectbox("Performance metric", metrics)
            st.altair_chart(model_metric_chart(comparison, metric), width="stretch")
        st.success(f"Best saved holdout model: **{best_model_name}**")
        if not cv_results.empty:
            with st.expander("Cross-validation results"):
                st.dataframe(cv_results, width="stretch", hide_index=True)
        if not diagnostics.empty:
            with st.expander("Model diagnostics"):
                st.dataframe(diagnostics, width="stretch", hide_index=True)

with insights_tab:
    st.subheader("Drivers and business insights")
    smoker_means = data.groupby("smoker")["charges"].mean()
    c = st.columns(3)
    c[0].metric("Rows after cleaning", f"{len(data):,}")
    c[1].metric("Mean charges — smokers", format_currency(float(smoker_means.get("yes", float("nan")))))
    c[2].metric("Mean charges — non-smokers", format_currency(float(smoker_means.get("no", float("nan")))))
    st.markdown("""
    - Smoking status is strongly associated with higher observed charges in this dataset.
    - Age and BMI show meaningful relationships with insurance charges.
    - Nonlinear models outperform the linear baseline on the saved holdout set.
    - Findings are predictive/associational and do not establish causation.
    """)
    if not importance.empty and {"Feature", "importance"}.issubset(importance.columns):
        st.markdown("#### Decision Tree feature importance")
        st.altair_chart(feature_importance_chart(importance), width="stretch")
        st.dataframe(importance.head(15), width="stretch", hide_index=True)
    if not segments.empty:
        st.markdown("#### Predicted-cost segmentation")
        st.dataframe(segments, width="stretch", hide_index=True)

with explorer_tab:
    st.subheader("Filtered dataset explorer")
    if filtered_data.empty:
        st.warning("No rows match the current filters.")
    else:
        c = st.columns(4)
        c[0].metric("Records", f"{len(filtered_data):,}")
        c[1].metric("Average charge", format_currency(float(filtered_data.charges.mean())))
        c[2].metric("Average age", f"{filtered_data.age.mean():.1f}")
        c[3].metric("Average BMI", f"{filtered_data.bmi.mean():.1f}")
        st.dataframe(filtered_data, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download filtered CSV",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="medical_insurance_filtered.csv",
            mime="text/csv",
        )
        with st.expander("Descriptive statistics"):
            st.dataframe(filtered_data.describe(include="all").transpose(), width="stretch")

with files_tab:
    st.subheader("Project files and reproducibility")
    st.markdown("Complete portfolio workflow: notebook, reusable source files, three saved model pipelines, reports, visualizations, documentation, and Streamlit app.")
    c1, c2 = st.columns(2)
    with c1:
        if REPORT_MD_PATH.exists():
            st.download_button("⬇️ Project report (Markdown)", REPORT_MD_PATH.read_text(encoding="utf-8"), "project_report.md", "text/markdown", width="stretch")
    with c2:
        if REPORT_PDF_PATH.exists():
            st.download_button("⬇️ Project report (PDF)", REPORT_PDF_PATH.read_bytes(), "project_report.pdf", "application/pdf", width="stretch")
    st.markdown("""
    **Reproducibility controls**
    - target `charges` is excluded from the feature matrix
    - preprocessing stays inside scikit-learn pipelines
    - tuning uses training data only
    - the final test set is reserved for final evaluation
    - `random_state=42` is used where applicable
    """)

st.divider()
st.caption("Medical Insurance Charges Regression • Linear vs Polynomial vs Decision Tree • Portfolio project by Alok Agarwal")