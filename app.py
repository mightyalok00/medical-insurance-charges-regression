"""Professional Streamlit dashboard for the Medical Insurance Charges Regression project."""

from pathlib import Path
from typing import Dict

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
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
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.06);
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 14px;
            padding: 0.9rem;
        }

        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(120, 120, 120, 0.20);
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(31, 119, 180, 0.11),
                rgba(44, 160, 44, 0.06)
            );
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }

        .hero p {
            margin: 0;
            opacity: 0.82;
        }

        .section-note {
            padding: 0.8rem 1rem;
            border-left: 4px solid #4c78a8;
            background: rgba(76, 120, 168, 0.06);
            border-radius: 8px;
            margin: 0.5rem 0 1rem 0;
        }

        .small-muted {
            font-size: 0.9rem;
            opacity: 0.72;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Load a CSV when present; otherwise return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return clean_columns(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def normalize_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature-importance exports for robust charting."""
    if df.empty:
        return df

    cleaned = clean_columns(df)

    if "Unnamed: 0" in cleaned.columns:
        cleaned = cleaned.rename(columns={"Unnamed: 0": "Feature"})

    feature_aliases = ["feature", "features", "Feature Name", "Feature_Name"]
    if "Feature" not in cleaned.columns:
        for alias in feature_aliases:
            if alias in cleaned.columns:
                cleaned = cleaned.rename(columns={alias: "Feature"})
                break

    importance_aliases = [
        "Importance",
        "feature_importance",
        "Feature Importance",
        "Feature_Importance",
    ]
    if "importance" not in cleaned.columns:
        for alias in importance_aliases:
            if alias in cleaned.columns:
                cleaned = cleaned.rename(columns={alias: "importance"})
                break

    unnamed = [
        col for col in cleaned.columns if str(col).lower().startswith("unnamed:")
    ]
    return cleaned.drop(columns=unnamed, errors="ignore")


@st.cache_resource
def load_models() -> Dict[str, object]:
    """Load all trained model pipelines available in the project."""
    models = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            continue

        try:
            models[model_name] = joblib.load(model_path)
        except Exception:
            continue

    return models


@st.cache_data
def load_project_assets():
    """Load the cleaned dataset and analysis exports."""
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
    importance = normalize_feature_importance(
        read_csv_if_exists(FEATURE_IMPORTANCE_PATH)
    )

    return data, comparison, diagnostics, cv_results, segments, importance


def first_existing_column(df: pd.DataFrame, candidates: list[str]):
    """Return the first column name present from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def format_currency(value: float) -> str:
    """Format numeric values as US-dollar currency."""
    return f"${value:,.2f}"


def model_metric_chart(df: pd.DataFrame, metric: str):
    """Create an explicit Altair chart for model comparison."""
    plot_df = df[["Model", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna()

    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X("Model:N", title=None, sort=None),
            y=alt.Y(f"{metric}:Q", title=metric),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=",.3f"),
            ],
        )
        .properties(height=360)
    )


def feature_importance_chart(df: pd.DataFrame):
    """Create a horizontal chart for Decision Tree feature importance."""
    plot_df = df[["Feature", "importance"]].copy()
    plot_df["importance"] = pd.to_numeric(plot_df["importance"], errors="coerce")
    plot_df = (
        plot_df.dropna()
        .sort_values("importance", ascending=False)
        .head(12)
    )

    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("importance:Q", title="Feature importance"),
            y=alt.Y("Feature:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=390)
    )


def charge_distribution_chart(df: pd.DataFrame):
    """Create a histogram of observed charges."""
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.82)
        .encode(
            x=alt.X("charges:Q", bin=alt.Bin(maxbins=35), title="Observed charges"),
            y=alt.Y("count():Q", title="Records"),
            tooltip=[alt.Tooltip("count():Q", title="Records")],
        )
        .properties(height=330)
    )


def smoker_charge_chart(df: pd.DataFrame):
    """Create a boxplot for smoker versus non-smoker charges."""
    return (
        alt.Chart(df)
        .mark_boxplot(size=48)
        .encode(
            x=alt.X("smoker:N", title="Smoker status"),
            y=alt.Y("charges:Q", title="Charges"),
            tooltip=[alt.Tooltip("smoker:N", title="Smoker status")],
        )
        .properties(height=330)
    )


# ---------------------------------------------------------------------
# Load assets
# ---------------------------------------------------------------------
models = load_models()
data, comparison, diagnostics, cv_results, segments, importance = load_project_assets()

required_columns = {
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
}
missing_columns = required_columns - set(data.columns)

if missing_columns:
    st.error(
        "The cleaned dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# ---------------------------------------------------------------------
# Dataset bounds and shared values
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.markdown("## 🏥 Project Navigator")
st.sidebar.caption("Medical Insurance Charges Regression")

st.sidebar.markdown("### 🎛️ Data filters")
age_filter = st.sidebar.slider(
    "🎂 Age",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
)
bmi_filter = st.sidebar.slider(
    "⚖️ BMI",
    min_value=round(bmi_min, 1),
    max_value=round(bmi_max, 1),
    value=(round(bmi_min, 1), round(bmi_max, 1)),
    step=0.1,
)
children_filter = st.sidebar.slider(
    "👶 Children",
    min_value=children_min,
    max_value=children_max,
    value=(children_min, children_max),
)
sex_filter = st.sidebar.multiselect("🚻 Sex", sex_options, default=sex_options)
smoker_filter = st.sidebar.multiselect(
    "🚬 Smoker status",
    smoker_options,
    default=smoker_options,
)
region_filter = st.sidebar.multiselect(
    "📍 Region",
    region_options,
    default=region_options,
)

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

if st.sidebar.button("↺ Reset filters", width="stretch"):
    st.rerun()

st.sidebar.markdown(
    """
    <div class="small-muted">
    Portfolio project • 3 regression models • leakage-safe validation
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏥 Medical Insurance Charges Regression</h1>
        <p>
            End-to-end machine learning portfolio dashboard comparing
            Linear Regression, Polynomial Regression, and Decision Tree Regression.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(4)
status_cols[0].metric("Dataset rows", f"{len(data):,}")
status_cols[1].metric("Trained models", f"{len(models)}/3")
status_cols[2].metric("Median charge", format_currency(median_charge))
status_cols[3].metric("Filtered records", f"{len(filtered_data):,}")

st.caption(
    "Educational portfolio project. Predictions are descriptive ML outputs and are "
    "not intended for real insurance underwriting, pricing, eligibility, or adverse-action decisions."
)


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------
with overview_tab:
    st.subheader("Executive overview")

    if filtered_data.empty:
        st.warning("No records match the selected sidebar filters.")
    else:
        avg_charge = float(filtered_data["charges"].mean())
        avg_age = float(filtered_data["age"].mean())
        avg_bmi = float(filtered_data["bmi"].mean())
        smoker_share = (
            filtered_data["smoker"]
            .astype(str)
            .str.lower()
            .eq("yes")
            .mean()
            * 100
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Average charge", format_currency(avg_charge))
        metric_cols[1].metric("Average age", f"{avg_age:.1f}")
        metric_cols[2].metric("Average BMI", f"{avg_bmi:.1f}")
        metric_cols[3].metric("Smoker share", f"{smoker_share:.1f}%")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Charge distribution")
            st.altair_chart(
                charge_distribution_chart(filtered_data),
                width="stretch",
            )

        with right:
            st.markdown("#### Charges by smoking status")
            st.altair_chart(
                smoker_charge_chart(filtered_data),
                width="stretch",
            )

    st.markdown(
        """
        <div class="section-note">
        <strong>Portfolio summary.</strong>
        The project evaluates three regression approaches on the same target,
        uses training-only cross-validation for model selection, and preserves
        an untouched test set for final generalization assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Prediction Lab
# ---------------------------------------------------------------------
with prediction_tab:
    st.subheader("Prediction lab")
    st.caption(
        "Enter one customer profile and compare predictions from every trained model."
    )

    if not models:
        st.error("No trained model files could be loaded from the models/ directory.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                pred_age = st.number_input(
                    "🎂 Age",
                    min_value=age_min,
                    max_value=age_max,
                    value=int(data["age"].median()),
                    step=1,
                )
                pred_sex = st.selectbox("🚻 Sex", sex_options)

            with col2:
                pred_bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=round(bmi_min, 1),
                    max_value=round(bmi_max, 1),
                    value=round(float(data["bmi"].median()), 1),
                    step=0.1,
                )
                pred_children = st.number_input(
                    "👶 Children",
                    min_value=children_min,
                    max_value=children_max,
                    value=int(data["children"].median()),
                    step=1,
                )

            with col3:
                pred_smoker = st.selectbox("🚬 Smoker", smoker_options)
                pred_region = st.selectbox("📍 Region", region_options)

            submitted = st.form_submit_button(
                "Generate model predictions",
                type="primary",
                width="stretch",
            )

        if submitted:
            input_df = pd.DataFrame(
                [
                    {
                        "age": pred_age,
                        "sex": pred_sex,
                        "bmi": pred_bmi,
                        "children": pred_children,
                        "smoker": pred_smoker,
                        "region": pred_region,
                    }
                ]
            )

            prediction_rows = []

            for model_name, model in models.items():
                try:
                    prediction_value = float(model.predict(input_df)[0])
                    prediction_rows.append(
                        {
                            "Model": model_name,
                            "Predicted Charges": prediction_value,
                        }
                    )
                except Exception as exc:
                    st.warning(f"{model_name} prediction failed: {exc}")

            if prediction_rows:
                predictions = pd.DataFrame(prediction_rows)
                predictions["Predicted Charges"] = predictions[
                    "Predicted Charges"
                ].clip(lower=0)

                decision_tree_row = predictions[
                    predictions["Model"].eq("Decision Tree Regression")
                ]

                preferred_prediction = (
                    float(decision_tree_row["Predicted Charges"].iloc[0])
                    if not decision_tree_row.empty
                    else float(predictions["Predicted Charges"].median())
                )

                if preferred_prediction < q1_charge:
                    cost_band = "Lower-cost band"
                    band_icon = "🟢"
                elif preferred_prediction > q3_charge:
                    cost_band = "Higher-cost band"
                    band_icon = "🔴"
                else:
                    cost_band = "Mid-cost band"
                    band_icon = "🟡"

                result_cols = st.columns(3)
                result_cols[0].metric(
                    "Primary model estimate",
                    format_currency(preferred_prediction),
                )
                result_cols[1].metric(
                    "Descriptive cost band",
                    f"{band_icon} {cost_band}",
                )
                result_cols[2].metric(
                    "Model spread",
                    format_currency(
                        float(
                            predictions["Predicted Charges"].max()
                            - predictions["Predicted Charges"].min()
                        )
                    ),
                )

                st.markdown("#### Cross-model prediction comparison")
                display_predictions = predictions.copy()
                display_predictions["Predicted Charges"] = display_predictions[
                    "Predicted Charges"
                ].map(format_currency)
                st.dataframe(
                    display_predictions,
                    width="stretch",
                    hide_index=True,
                )

                pred_chart_df = predictions.copy()
                pred_chart = (
                    alt.Chart(pred_chart_df)
                    .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
                    .encode(
                        x=alt.X("Model:N", title=None, sort=None),
                        y=alt.Y(
                            "Predicted Charges:Q",
                            title="Predicted charges",
                        ),
                        tooltip=[
                            alt.Tooltip("Model:N", title="Model"),
                            alt.Tooltip(
                                "Predicted Charges:Q",
                                title="Prediction",
                                format=",.2f",
                            ),
                        ],
                    )
                    .properties(height=330)
                )
                st.altair_chart(pred_chart, width="stretch")

                st.info(
                    "The cost band is descriptive only. It is based on observed "
                    "charge quartiles in this dataset and is not an actuarial risk classification."
                )


# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
with performance_tab:
    st.subheader("Model performance")
    st.caption(
        "Compare the same three regression models using saved evaluation results."
    )

    if comparison.empty:
        st.warning("Model comparison results are not available.")
    else:
        st.dataframe(comparison, width="stretch", hide_index=True)

        metric_candidates = [
            col
            for col in [
                "Test RMSE",
                "RMSE",
                "Test MAE",
                "MAE",
                "Test MSE",
                "MSE",
                "Test R2",
                "R2",
                "Test R²",
                "R²",
            ]
            if col in comparison.columns
        ]

        if "Model" in comparison.columns and metric_candidates:
            selected_metric = st.selectbox(
                "Performance metric",
                metric_candidates,
                key="performance_metric",
            )
            st.altair_chart(
                model_metric_chart(comparison, selected_metric),
                width="stretch",
            )

        rmse_col = first_existing_column(comparison, ["Test RMSE", "RMSE"])
        r2_col = first_existing_column(
            comparison,
            ["Test R2", "R2", "Test R²", "R²"],
        )

        summary_cols = st.columns(2)

        if rmse_col and "Model" in comparison.columns:
            rmse_values = pd.to_numeric(comparison[rmse_col], errors="coerce")
            if rmse_values.notna().any():
                idx = rmse_values.idxmin()
                summary_cols[0].metric(
                    "Lowest test RMSE",
                    str(comparison.loc[idx, "Model"]),
                    delta=f"{float(rmse_values.loc[idx]):,.2f}",
                    delta_color="off",
                )

        if r2_col and "Model" in comparison.columns:
            r2_values = pd.to_numeric(comparison[r2_col], errors="coerce")
            if r2_values.notna().any():
                idx = r2_values.idxmax()
                summary_cols[1].metric(
                    "Highest test R²",
                    str(comparison.loc[idx, "Model"]),
                    delta=f"{float(r2_values.loc[idx]):.3f}",
                    delta_color="off",
                )

    if not diagnostics.empty:
        with st.expander("Generalization diagnostics", expanded=False):
            st.dataframe(diagnostics, width="stretch", hide_index=True)

    if not cv_results.empty:
        with st.expander("Cross-validation results", expanded=False):
            st.dataframe(cv_results, width="stretch", hide_index=True)


# ---------------------------------------------------------------------
# Drivers & Insights
# ---------------------------------------------------------------------
with insights_tab:
    st.subheader("Drivers and business insights")

    smoker_group = data.groupby("smoker")["charges"].agg(["mean", "median", "count"])

    smoker_yes = float(
        smoker_group.loc["yes", "mean"]
        if "yes" in smoker_group.index
        else float("nan")
    )
    smoker_no = float(
        smoker_group.loc["no", "mean"]
        if "no" in smoker_group.index
        else float("nan")
    )

    insight_cols = st.columns(4)
    insight_cols[0].metric("Cleaned records", f"{len(data):,}")
    insight_cols[1].metric("Mean charge — smokers", format_currency(smoker_yes))
    insight_cols[2].metric("Mean charge — non-smokers", format_currency(smoker_no))
    insight_cols[3].metric(
        "Observed smoker uplift",
        (
            f"{((smoker_yes / smoker_no) - 1) * 100:.1f}%"
            if smoker_no > 0
            else "N/A"
        ),
    )

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Decision Tree feature importance")
        if {
            "Feature",
            "importance",
        }.issubset(importance.columns):
            st.altair_chart(
                feature_importance_chart(importance),
                width="stretch",
            )
        else:
            st.info("Feature-importance export is unavailable.")

    with right:
        st.markdown("#### Interpretation")
        st.markdown(
            """
            - **Smoking status** is strongly associated with higher observed charges.
            - **Age and BMI** show meaningful relationships with medical charges.
            - **Polynomial and tree-based models** capture nonlinear structure that the linear baseline can miss.
            - **Region, sex, and children** can contribute to prediction, but their observed differences are smaller than the smoking effect in this dataset.
            - Model outputs are **predictive and associational**, not causal.
            """
        )

        if not segments.empty:
            st.markdown("#### Predicted-cost segments")
            st.dataframe(
                segments,
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------
with explorer_tab:
    st.subheader("Filtered data explorer")

    if filtered_data.empty:
        st.warning("No records match the selected filters.")
    else:
        explorer_cols = st.columns(4)
        explorer_cols[0].metric("Records", f"{len(filtered_data):,}")
        explorer_cols[1].metric(
            "Average charge",
            format_currency(float(filtered_data["charges"].mean())),
        )
        explorer_cols[2].metric(
            "Average age",
            f"{filtered_data['age'].mean():.1f}",
        )
        explorer_cols[3].metric(
            "Average BMI",
            f"{filtered_data['bmi'].mean():.1f}",
        )

        st.dataframe(
            filtered_data,
            width="stretch",
            hide_index=True,
            height=420,
        )

        with st.expander("Descriptive statistics", expanded=False):
            st.dataframe(
                filtered_data.describe(include="all").transpose(),
                width="stretch",
            )

        st.download_button(
            "⬇️ Download filtered dataset",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name="filtered_insurance_data.csv",
            mime="text/csv",
            width="stretch",
        )


# ---------------------------------------------------------------------
# Project Files
# ---------------------------------------------------------------------
with docs_tab:
    st.subheader("Project documentation")

    doc_cols = st.columns(3)
    doc_cols[0].metric("Regression models", "3")
    doc_cols[1].metric("Assignment questions", "17")
    doc_cols[2].metric("Validation approach", "5-fold CV")

    st.markdown(
        """
        **Project workflow**

        1. Inspect and clean the insurance dataset.
        2. Separate predictors from the `charges` target.
        3. Create an untouched train/test split.
        4. Keep encoding and preprocessing inside scikit-learn pipelines.
        5. Train Linear, Polynomial, and Decision Tree regression models.
        6. Use training-only cross-validation for model selection and tuning.
        7. Evaluate the final models on the same untouched test data.
        8. Translate results into business insights while documenting limitations.
        """
    )

    if REPORT_MD_PATH.exists():
        st.download_button(
            "⬇️ Download project report (Markdown)",
            data=REPORT_MD_PATH.read_bytes(),
            file_name="project_report.md",
            mime="text/markdown",
            width="stretch",
        )

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
        <div class="section-note">
        <strong>Responsible-use note.</strong>
        This dashboard is a portfolio demonstration built on a small public-style
        tabular dataset. It should not be deployed for real insurance pricing,
        underwriting, eligibility, or adverse-action workflows without actuarial,
        legal, fairness, temporal, external, and production validation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "Medical Insurance Charges Regression • Streamlit portfolio dashboard • "
    "Linear Regression • Polynomial Regression • Decision Tree Regression"
)
