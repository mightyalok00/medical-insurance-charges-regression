"""Professional Streamlit dashboard for the Medical Insurance Charges Regression project."""

from pathlib import Path
from typing import Dict

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance ML Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.06);
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 14px;
            padding: 0.9rem;
        }
        .hero {
            padding: 1.35rem 1.45rem;
            border: 1px solid rgba(120, 120, 120, 0.20);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(31,119,180,.11), rgba(44,160,44,.06));
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0 0 0.4rem 0; font-size: 2rem; }
        .hero p { margin: 0; opacity: 0.82; }
        .section-note {
            padding: 0.85rem 1rem;
            border-left: 4px solid #4c78a8;
            background: rgba(76,120,168,.06);
            border-radius: 8px;
            margin: 0.5rem 0 1rem 0;
        }
        .best-model {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(46,125,50,.30);
            background: rgba(46,125,50,.07);
            border-radius: 12px;
            margin: 0.5rem 0 1rem 0;
        }
        .small-muted { font-size: 0.9rem; opacity: 0.72; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


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

    cleaned = clean_columns(df)

    if "Unnamed: 0" in cleaned.columns:
        cleaned = cleaned.rename(columns={"Unnamed: 0": "Feature"})

    for alias in ["feature", "features", "Feature Name", "Feature_Name"]:
        if "Feature" not in cleaned.columns and alias in cleaned.columns:
            cleaned = cleaned.rename(columns={alias: "Feature"})

    for alias in ["Importance", "feature_importance", "Feature Importance", "Feature_Importance"]:
        if "importance" not in cleaned.columns and alias in cleaned.columns:
            cleaned = cleaned.rename(columns={alias: "importance"})

    unnamed = [c for c in cleaned.columns if str(c).lower().startswith("unnamed:")]
    return cleaned.drop(columns=unnamed, errors="ignore")


@st.cache_resource
def load_models() -> Dict[str, object]:
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
            except Exception:
                pass
    return models


@st.cache_data
def load_project_assets():
    if not DATA_PATH.exists():
        st.error(f"Required cleaned dataset was not found: {DATA_PATH}")
        st.stop()

    try:
        data = clean_columns(pd.read_csv(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load the cleaned dataset: {exc}")
        st.stop()

    comparison = read_csv_if_exists(COMPARISON_PATH)
    diagnostics = read_csv_if_exists(DIAGNOSTICS_PATH)
    cv_results = read_csv_if_exists(CV_PATH)
    segments = read_csv_if_exists(SEGMENTS_PATH)
    importance = normalize_feature_importance(read_csv_if_exists(FEATURE_IMPORTANCE_PATH))
    return data, comparison, diagnostics, cv_results, segments, importance


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def get_best_model_name(comparison_df: pd.DataFrame) -> str:
    if comparison_df.empty or "Model" not in comparison_df.columns:
        return "Decision Tree Regression"
    rmse_col = "Test RMSE" if "Test RMSE" in comparison_df.columns else None
    if rmse_col is None:
        return "Decision Tree Regression"
    values = pd.to_numeric(comparison_df[rmse_col], errors="coerce")
    if values.notna().any():
        return str(comparison_df.loc[values.idxmin(), "Model"])
    return "Decision Tree Regression"


def model_metric_chart(df: pd.DataFrame, metric: str):
    plot_df = df[["Model", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna()

    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X(field="Model", type="nominal", title=None, sort=None),
            y=alt.Y(field=metric, type="quantitative", title=metric),
            tooltip=[
                alt.Tooltip(field="Model", type="nominal", title="Model"),
                alt.Tooltip(field=metric, type="quantitative", title=metric, format=",.3f"),
            ],
        )
        .properties(height=360)
    )


def feature_importance_chart(df: pd.DataFrame):
    plot_df = df[["Feature", "importance"]].copy()
    plot_df["importance"] = pd.to_numeric(plot_df["importance"], errors="coerce")
    plot_df = plot_df.dropna().sort_values("importance", ascending=False).head(12)

    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(field="importance", type="quantitative", title="Feature importance"),
            y=alt.Y(field="Feature", type="nominal", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip(field="Feature", type="nominal", title="Feature"),
                alt.Tooltip(field="importance", type="quantitative", title="Importance", format=".4f"),
            ],
        )
        .properties(height=390)
    )


def charge_distribution_chart(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.82)
        .encode(
            x=alt.X(field="charges", type="quantitative", bin=alt.Bin(maxbins=35), title="Observed charges"),
            y=alt.Y("count():Q", title="Records"),
            tooltip=[alt.Tooltip("count():Q", title="Records")],
        )
        .properties(height=330)
    )


def smoker_charge_chart(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_boxplot(size=48)
        .encode(
            x=alt.X(field="smoker", type="nominal", title="Smoker status"),
            y=alt.Y(field="charges", type="quantitative", title="Charges"),
        )
        .properties(height=330)
    )


# -----------------------------------------------------------------------------
# Load project assets
# -----------------------------------------------------------------------------
models = load_models()
data, comparison, diagnostics, cv_results, segments, importance = load_project_assets()

required_columns = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
missing_columns = required_columns - set(data.columns)
if missing_columns:
    st.error("The cleaned dataset is missing required columns: " + ", ".join(sorted(missing_columns)))
    st.stop()

best_model_name = get_best_model_name(comparison)

age_min = int(data["age"].min())
age_max = int(data["age"].max())
bmi_min = float(data["bmi"].min())
bmi_max = float(data["bmi"].max())
children_min = int(data["children"].min())
children_max = int(data["children"].max())

sex_options = sorted(data["sex"].dropna().astype(str).unique().tolist())
smoker_options = sorted(data["smoker"].dropna().astype(str).unique().tolist())
region_options = sorted(data["region"].dropna().astype(str).unique().tolist())

q1_charge = float(data["charges"].quantile(0.25))
median_charge = float(data["charges"].median())
q3_charge = float(data["charges"].quantile(0.75))


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🏥 Project Navigator")
st.sidebar.caption("Medical Insurance Charges Regression")

st.sidebar.markdown("### 🎛️ Dataset filters")
age_filter = st.sidebar.slider("🎂 Age", age_min, age_max, (age_min, age_max))
bmi_filter = st.sidebar.slider(
    "⚖️ BMI",
    min_value=round(bmi_min, 1),
    max_value=round(bmi_max, 1),
    value=(round(bmi_min, 1), round(bmi_max, 1)),
    step=0.1,
)
children_filter = st.sidebar.slider(
    "👶 Children", children_min, children_max, (children_min, children_max)
)
sex_filter = st.sidebar.multiselect("🚻 Sex", sex_options, default=sex_options)
smoker_filter = st.sidebar.multiselect("🚬 Smoker status", smoker_options, default=smoker_options)
region_filter = st.sidebar.multiselect("📍 Region", region_options, default=region_options)

filtered_data = data[
    data["age"].between(*age_filter)
    & data["bmi"].between(*bmi_filter)
    & data["children"].between(*children_filter)
    & data["sex"].astype(str).isin(sex_filter)
    & data["smoker"].astype(str).isin(smoker_filter)
    & data["region"].astype(str).isin(region_filter)
].copy()

st.sidebar.divider()
st.sidebar.metric("Filtered records", f"{len(filtered_data):,}")
st.sidebar.caption(f"Current best saved test model: **{best_model_name}**")

if st.sidebar.button("↺ Reset filters", width="stretch"):
    st.rerun()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏥 Medical Insurance Charges Regression</h1>
        <p>
            End-to-end machine learning portfolio dashboard comparing Linear Regression,
            Polynomial Regression, and Decision Tree Regression.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(5)
status_cols[0].metric("Dataset rows", f"{len(data):,}")
status_cols[1].metric("Trained models", f"{len(models)}/3")
status_cols[2].metric("Best test model", best_model_name.replace(" Regression", ""))
status_cols[3].metric("Median charge", format_currency(median_charge))
status_cols[4].metric("Filtered records", f"{len(filtered_data):,}")

st.caption(
    "Educational portfolio project. Predictions are descriptive ML outputs and are not intended "
    "for real insurance underwriting, pricing, eligibility, or adverse-action decisions."
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
overview_tab, prediction_tab, performance_tab, insights_tab, explorer_tab, docs_tab = st.tabs(
    [
        "🏠 Executive Overview",
        "🔮 Prediction Lab",
        "📊 Model Performance",
        "🧠 Drivers & Insights",
        "🔎 Data Explorer",
        "📄 Project Files",
    ]
)


# -----------------------------------------------------------------------------
# Executive overview
# -----------------------------------------------------------------------------
with overview_tab:
    st.subheader("Executive overview")

    if filtered_data.empty:
        st.warning("No records match the selected sidebar filters.")
    else:
        avg_charge = float(filtered_data["charges"].mean())
        avg_age = float(filtered_data["age"].mean())
        avg_bmi = float(filtered_data["bmi"].mean())
        smoker_share = filtered_data["smoker"].astype(str).str.lower().eq("yes").mean() * 100

        metric_cols = st.columns(4)
        metric_cols[0].metric("Average charge", format_currency(avg_charge))
        metric_cols[1].metric("Average age", f"{avg_age:.1f}")
        metric_cols[2].metric("Average BMI", f"{avg_bmi:.1f}")
        metric_cols[3].metric("Smoker share", f"{smoker_share:.1f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Charge distribution")
            st.altair_chart(charge_distribution_chart(filtered_data), width="stretch")
        with right:
            st.markdown("#### Charges by smoking status")
            st.altair_chart(smoker_charge_chart(filtered_data), width="stretch")

    st.markdown(
        f"""
        <div class="best-model">
        <strong>Best saved test model:</strong> {best_model_name}. The selection is based on the
        lowest saved holdout Test RMSE in <code>reports/model_comparison.csv</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Prediction lab
# -----------------------------------------------------------------------------
with prediction_tab:
    st.subheader("Prediction lab")
    st.write(
        "Choose a trained model or compare all three models using the same customer profile. "
        "The selector makes it easy to inspect how model choice changes the predicted charge."
    )

    available_model_names = [name for name in MODEL_PATHS if name in models]
    prediction_mode_options = ["Compare all models"] + available_model_names

    selected_model = st.selectbox(
        "🧠 Select prediction model",
        options=prediction_mode_options,
        index=0,
        help="Compare all saved models or run a prediction using one selected model.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", age_min, age_max, int(data["age"].median()))
        sex = st.selectbox("Sex", sex_options)
    with c2:
        bmi = st.slider(
            "BMI",
            min_value=round(bmi_min, 1),
            max_value=round(bmi_max, 1),
            value=round(float(data["bmi"].median()), 1),
            step=0.1,
        )
        children = st.slider(
            "Children", children_min, children_max, int(data["children"].median())
        )
    with c3:
        smoker = st.selectbox("Smoker", smoker_options)
        region = st.selectbox("Region", region_options)

    profile = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }]
    )

    if st.button("💰 Generate prediction", type="primary", width="stretch"):
        if not models:
            st.error("No trained model artifacts could be loaded.")
        else:
            try:
                if selected_model == "Compare all models":
                    predictions = []
                    for model_name in available_model_names:
                        prediction = float(models[model_name].predict(profile)[0])
                        predictions.append({"Model": model_name, "Predicted Charge": prediction})

                    prediction_df = pd.DataFrame(predictions)
                    prediction_df["Predicted Charge"] = prediction_df["Predicted Charge"].round(2)
                    prediction_df["Difference vs lowest"] = (
                        prediction_df["Predicted Charge"] - prediction_df["Predicted Charge"].min()
                    ).round(2)

                    st.markdown("#### Three-model prediction comparison")
                    st.dataframe(
                        prediction_df.style.format({
                            "Predicted Charge": "${:,.2f}",
                            "Difference vs lowest": "${:,.2f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

                    chart_df = prediction_df.copy()
                    comparison_chart = (
                        alt.Chart(chart_df)
                        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
                        .encode(
                            x=alt.X(field="Model", type="nominal", title=None, sort=None),
                            y=alt.Y(field="Predicted Charge", type="quantitative", title="Predicted charge"),
                            tooltip=[
                                alt.Tooltip(field="Model", type="nominal"),
                                alt.Tooltip(field="Predicted Charge", type="quantitative", format="$,.2f"),
                            ],
                        )
                        .properties(height=340)
                    )
                    st.altair_chart(comparison_chart, width="stretch")

                    if best_model_name in prediction_df["Model"].values:
                        best_prediction = float(
                            prediction_df.loc[
                                prediction_df["Model"] == best_model_name,
                                "Predicted Charge",
                            ].iloc[0]
                        )
                        st.success(
                            f"Portfolio default: **{best_model_name}** → {format_currency(best_prediction)}"
                        )
                else:
                    prediction = float(models[selected_model].predict(profile)[0])
                    st.metric(f"Predicted charge — {selected_model}", format_currency(prediction))

                    if prediction < q1_charge:
                        band = "Lower-cost band"
                    elif prediction > q3_charge:
                        band = "Higher-cost band"
                    else:
                        band = "Mid-cost band"

                    st.info(
                        f"Descriptive predicted-cost segment: **{band}**. "
                        "This is not an actuarial or underwriting risk classification."
                    )

                    if selected_model == best_model_name:
                        st.success("This is the strongest saved holdout-test model in the project.")
                    else:
                        st.caption(
                            f"Saved holdout-test comparison currently identifies **{best_model_name}** "
                            "as the lowest-RMSE model."
                        )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# -----------------------------------------------------------------------------
# Model performance
# -----------------------------------------------------------------------------
with performance_tab:
    st.subheader("Model performance")

    if comparison.empty:
        st.warning("Model comparison report is unavailable.")
    else:
        st.dataframe(comparison, width="stretch", hide_index=True)

        metric_options = [
            col for col in ["Test RMSE", "Test MAE", "Test MSE", "Test R2", "Test R²"]
            if col in comparison.columns
        ]
        if "Model" in comparison.columns and metric_options:
            selected_metric = st.selectbox("Performance metric", metric_options)
            st.altair_chart(model_metric_chart(comparison, selected_metric), width="stretch")

        if "Test RMSE" in comparison.columns and "Model" in comparison.columns:
            rmse = pd.to_numeric(comparison["Test RMSE"], errors="coerce")
            if rmse.notna().any():
                row = comparison.loc[rmse.idxmin()]
                st.success(
                    f"Best saved holdout result: **{row['Model']}** with Test RMSE "
                    f"**{float(row['Test RMSE']):,.2f}**."
                )

        if not cv_results.empty:
            with st.expander("Cross-validation results"):
                st.dataframe(cv_results, width="stretch", hide_index=True)

        if not diagnostics.empty:
            with st.expander("Model diagnostics"):
                st.dataframe(diagnostics, width="stretch", hide_index=True)

    st.markdown(
        """
        **How to interpret the models**

        - **Linear Regression:** transparent baseline and easiest to explain.
        - **Polynomial Regression:** captures nonlinear relationships while retaining a regression form.
        - **Decision Tree Regression:** strongest saved test performance and naturally captures thresholds and interactions.
        """
    )


# -----------------------------------------------------------------------------
# Drivers and insights
# -----------------------------------------------------------------------------
with insights_tab:
    st.subheader("Drivers and business insights")

    smoker_means = data.groupby("smoker")["charges"].mean()
    insight_cols = st.columns(3)
    insight_cols[0].metric("Rows after cleaning", f"{len(data):,}")
    insight_cols[1].metric(
        "Mean charges — smokers",
        format_currency(float(smoker_means.get("yes", float("nan")))),
    )
    insight_cols[2].metric(
        "Mean charges — non-smokers",
        format_currency(float(smoker_means.get("no", float("nan")))),
    )

    st.markdown(
        """
        - Smoking status is strongly associated with higher observed charges in this dataset.
        - Age and BMI show meaningful relationships with insurance charges.
        - The improvement of nonlinear models over Linear Regression suggests interaction and threshold effects.
        - Region, sex, and number of children show smaller descriptive differences than smoking status in this sample.
        - These results are predictive/associational and do not establish causation.
        """
    )

    if not importance.empty and {"Feature", "importance"}.issubset(importance.columns):
        st.markdown("#### Decision Tree feature importance")
        st.altair_chart(feature_importance_chart(importance), width="stretch")
        st.dataframe(importance.head(15), width="stretch", hide_index=True)

    if not segments.empty:
        st.markdown("#### Predicted-cost segmentation")
        st.dataframe(segments, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Data explorer
# -----------------------------------------------------------------------------
with explorer_tab:
    st.subheader("Filtered dataset explorer")

    if filtered_data.empty:
        st.warning("No rows match the current sidebar filters.")
    else:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Records", f"{len(filtered_data):,}")
        summary_cols[1].metric("Average charge", format_currency(float(filtered_data["charges"].mean())))
        summary_cols[2].metric("Average age", f"{filtered_data['age'].mean():.1f}")
        summary_cols[3].metric("Average BMI", f"{filtered_data['bmi'].mean():.1f}")

        st.dataframe(filtered_data, width="stretch", hide_index=True)

        csv_bytes = filtered_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered CSV",
            data=csv_bytes,
            file_name="medical_insurance_filtered.csv",
            mime="text/csv",
        )

        with st.expander("Descriptive statistics"):
            st.dataframe(filtered_data.describe(include="all").transpose(), width="stretch")


# -----------------------------------------------------------------------------
# Project files
# -----------------------------------------------------------------------------
with docs_tab:
    st.subheader("Project files and reproducibility")

    st.markdown(
        """
        The repository contains the complete portfolio workflow: executed notebook, reusable source files,
        three trained model pipelines, analysis reports, visualizations, professional documentation,
        and this Streamlit application.
        """
    )

    file_cols = st.columns(2)
    with file_cols[0]:
        if REPORT_MD_PATH.exists():
            st.download_button(
                "⬇️ Download project report (Markdown)",
                data=REPORT_MD_PATH.read_text(encoding="utf-8"),
                file_name="project_report.md",
                mime="text/markdown",
                width="stretch",
            )
    with file_cols[1]:
        if REPORT_PDF_PATH.exists():
            st.download_button(
                "⬇️ Download project report (PDF)",
                data=REPORT_PDF_PATH.read_bytes(),
                file_name="project_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

    st.markdown(
        """
        **Reproducibility controls**

        - target `charges` is excluded from the feature matrix
        - preprocessing remains inside scikit-learn pipelines
        - model selection/tuning uses training data only
        - final evaluation uses an untouched holdout test set
        - `random_state=42` is used where applicable
        """
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Medical Insurance Charges Regression • Linear vs Polynomial vs Decision Tree • "
    "Portfolio project by Alok Agarwal"
)
