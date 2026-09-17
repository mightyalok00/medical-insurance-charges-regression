"""Streamlit dashboard for the Medical Insurance Charges Regression project."""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "decision_tree_regression.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "insurance_cleaned.csv"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"
SEGMENTS_PATH = BASE_DIR / "reports" / "predicted_cost_segments.csv"
FEATURE_IMPORTANCE_PATH = (
    BASE_DIR / "reports" / "decision_tree_feature_importance.csv"
)


# -------------------------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Dashboard",
    page_icon="🏥",
    layout="wide",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove accidental whitespace from column names."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def clean_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature-importance CSVs saved with or without an index."""
    cleaned = clean_columns(df)

    # Common pandas export artifact when index=True was used.
    if "Unnamed: 0" in cleaned.columns:
        cleaned = cleaned.rename(columns={"Unnamed: 0": "Feature"})

    # Support alternate feature-name column labels.
    feature_aliases = ["feature", "features", "Feature Name", "Feature_Name"]
    if "Feature" not in cleaned.columns:
        for alias in feature_aliases:
            if alias in cleaned.columns:
                cleaned = cleaned.rename(columns={alias: "Feature"})
                break

    # Support alternate importance column labels.
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

    # If an unnamed column remains, remove it safely.
    unnamed = [
        col
        for col in cleaned.columns
        if str(col).lower().startswith("unnamed:")
    ]
    cleaned = cleaned.drop(columns=unnamed, errors="ignore")

    return cleaned


@st.cache_resource
def load_model():
    """Load the tuned Decision Tree pipeline once per Streamlit session."""
    if not MODEL_PATH.exists():
        st.error(f"❌ Trained model not found:\n{MODEL_PATH}")
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"❌ Could not load the trained model: {exc}")
        st.stop()


@st.cache_data
def load_project_data():
    """Load the cleaned dataset and saved analysis outputs."""
    required = {
        "Cleaned dataset": DATA_PATH,
        "Model comparison": COMPARISON_PATH,
    }

    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        st.error("❌ Required project files are missing:\n\n" + "\n".join(missing))
        st.stop()

    try:
        data = clean_columns(pd.read_csv(DATA_PATH))
        comparison = clean_columns(pd.read_csv(COMPARISON_PATH))

        segments = (
            clean_columns(pd.read_csv(SEGMENTS_PATH))
            if SEGMENTS_PATH.exists()
            else pd.DataFrame()
        )

        importance = (
            clean_feature_importance(pd.read_csv(FEATURE_IMPORTANCE_PATH))
            if FEATURE_IMPORTANCE_PATH.exists()
            else pd.DataFrame()
        )

        return data, comparison, segments, importance

    except Exception as exc:
        st.error(f"❌ Could not load project data: {exc}")
        st.stop()


def make_model_chart(df: pd.DataFrame, metric: str):
    """Create a robust Altair model-comparison chart."""
    plot_df = df[["Model", metric]].dropna().copy()

    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(
                field="Model",
                type="nominal",
                title="Model",
                sort=None,
            ),
            y=alt.Y(
                field=metric,
                type="quantitative",
                title=metric,
            ),
            tooltip=[
                alt.Tooltip(field="Model", type="nominal", title="Model"),
                alt.Tooltip(field=metric, type="quantitative", title=metric, format=",.3f"),
            ],
        )
        .properties(height=380)
    )


def make_importance_chart(df: pd.DataFrame):
    """Create a robust feature-importance chart without shorthand parsing."""
    plot_df = (
        df[["Feature", "importance"]]
        .dropna()
        .sort_values("importance", ascending=False)
        .head(12)
        .copy()
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                field="importance",
                type="quantitative",
                title="Importance",
            ),
            y=alt.Y(
                field="Feature",
                type="nominal",
                title="Feature",
                sort="-x",
            ),
            tooltip=[
                alt.Tooltip(field="Feature", type="nominal", title="Feature"),
                alt.Tooltip(
                    field="importance",
                    type="quantitative",
                    title="Importance",
                    format=".4f",
                ),
            ],
        )
        .properties(height=420)
    )


# -------------------------------------------------------------------
# Load resources
# -------------------------------------------------------------------
model = load_model()
data, comparison, segments, importance = load_project_data()


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("🏥 Medical Insurance Charges Regression Dashboard")

st.caption(
    "Portfolio dashboard using Linear Regression, Polynomial Regression, "
    "and Decision Tree Regression. Predictions are for educational "
    "demonstration only and should not be used for real insurance pricing, "
    "eligibility, or underwriting decisions."
)


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🎛️ Dashboard Filters")
st.sidebar.caption("Filter the cleaned insurance dataset interactively.")

age_min = int(data["age"].min())
age_max = int(data["age"].max())
bmi_min = float(data["bmi"].min())
bmi_max = float(data["bmi"].max())
child_min = int(data["children"].min())
child_max = int(data["children"].max())

age_range = st.sidebar.slider(
    "🎂 Age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
)

bmi_range = st.sidebar.slider(
    "⚖️ BMI range",
    min_value=float(round(bmi_min, 1)),
    max_value=float(round(bmi_max, 1)),
    value=(float(round(bmi_min, 1)), float(round(bmi_max, 1))),
    step=0.1,
)

children_range = st.sidebar.slider(
    "👶 Children range",
    min_value=child_min,
    max_value=child_max,
    value=(child_min, child_max),
)

sex_options = sorted(data["sex"].dropna().astype(str).unique().tolist())
smoker_options = sorted(data["smoker"].dropna().astype(str).unique().tolist())
region_options = sorted(data["region"].dropna().astype(str).unique().tolist())

sex_filter = st.sidebar.multiselect(
    "🚻 Sex",
    options=sex_options,
    default=sex_options,
)

smoker_filter = st.sidebar.multiselect(
    "🚬 Smoker status",
    options=smoker_options,
    default=smoker_options,
)

region_filter = st.sidebar.multiselect(
    "📍 Region",
    options=region_options,
    default=region_options,
)


filtered_data = data[
    data["age"].between(age_range[0], age_range[1])
    & data["bmi"].between(bmi_range[0], bmi_range[1])
    & data["children"].between(children_range[0], children_range[1])
    & data["sex"].astype(str).isin(sex_filter)
    & data["smoker"].astype(str).isin(smoker_filter)
    & data["region"].astype(str).isin(region_filter)
].copy()

st.sidebar.metric("🔎 Matching records", f"{len(filtered_data):,}")


# -------------------------------------------------------------------
# Dashboard tabs
# -------------------------------------------------------------------
predict_tab, performance_tab, insights_tab, data_tab = st.tabs(
    [
        "🔮 Predict Charges",
        "📊 Model Performance",
        "💡 Business Insights",
        "🗂️ Filtered Dataset",
    ]
)


# -------------------------------------------------------------------
# Prediction tab
# -------------------------------------------------------------------
with predict_tab:
    st.subheader("🔮 Estimate medical insurance charges")
    st.write(
        "Enter a customer profile below. The prediction uses the tuned "
        "Decision Tree Regression pipeline saved by the project."
    )

    left, middle, right = st.columns(3)

    with left:
        age = st.slider(
            "🎂 Age",
            min_value=age_min,
            max_value=age_max,
            value=int(data["age"].median()),
            step=1,
        )

        sex = st.selectbox(
            "🚻 Sex",
            options=sex_options,
        )

    with middle:
        bmi = st.slider(
            "⚖️ BMI",
            min_value=float(round(bmi_min, 1)),
            max_value=float(round(bmi_max, 1)),
            value=float(round(data["bmi"].median(), 1)),
            step=0.1,
        )

        children = st.slider(
            "👶 Number of children",
            min_value=child_min,
            max_value=child_max,
            value=int(data["children"].median()),
            step=1,
        )

    with right:
        smoker = st.selectbox(
            "🚬 Smoker",
            options=smoker_options,
        )

        region = st.selectbox(
            "📍 Region",
            options=region_options,
        )

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    if st.button(
        "💰 Predict insurance charges",
        type="primary",
        width="stretch",
    ):
        try:
            prediction = float(model.predict(input_df)[0])

            st.metric(
                "💵 Predicted annual charges",
                f"${prediction:,.2f}",
            )

            observed_q1 = float(data["charges"].quantile(0.25))
            observed_q3 = float(data["charges"].quantile(0.75))

            if prediction < observed_q1:
                band = "🟢 Lower-cost band"
            elif prediction > observed_q3:
                band = "🔴 Higher-cost band"
            else:
                band = "🟡 Mid-cost band"

            st.info(
                f"Descriptive cost segment: **{band}**. "
                "This is not an underwriting or actuarial risk classification."
            )

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")


# -------------------------------------------------------------------
# Model performance tab
# -------------------------------------------------------------------
with performance_tab:
    st.subheader("📊 Three-model comparison")

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
    )

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
            "📈 Chart metric",
            options=metric_candidates,
        )

        model_chart = make_model_chart(
            comparison,
            selected_metric,
        )

        st.altair_chart(
            model_chart,
            width="stretch",
        )

    else:
        st.warning(
            "⚠️ The model-comparison CSV does not contain the expected "
            "'Model' and metric columns."
        )

    rmse_col = None
    for candidate in ["Test RMSE", "RMSE"]:
        if candidate in comparison.columns:
            rmse_col = candidate
            break

    if rmse_col and "Model" in comparison.columns:
        numeric_rmse = pd.to_numeric(
            comparison[rmse_col],
            errors="coerce",
        )

        if numeric_rmse.notna().any():
            best_index = numeric_rmse.idxmin()
            best_row = comparison.loc[best_index]

            st.success(
                f"🏆 Lowest saved test RMSE: "
                f"**{best_row['Model']}** "
                f"({float(best_row[rmse_col]):,.2f})."
            )


# -------------------------------------------------------------------
# Business insights tab
# -------------------------------------------------------------------
with insights_tab:
    st.subheader("💡 Business and model insights")

    col1, col2, col3 = st.columns(3)

    smoker_means = data.groupby("smoker")["charges"].mean()

    with col1:
        st.metric(
            "🧹 Rows after cleaning",
            f"{len(data):,}",
        )

    with col2:
        smoker_mean = float(
            smoker_means.get("yes", float("nan"))
        )
        st.metric(
            "🚬 Mean charges — smokers",
            f"${smoker_mean:,.0f}",
        )

    with col3:
        nonsmoker_mean = float(
            smoker_means.get("no", float("nan"))
        )
        st.metric(
            "🚭 Mean charges — non-smokers",
            f"${nonsmoker_mean:,.0f}",
        )

    st.markdown(
        """
        **Interpretation notes**

        - 🚬 Smoking status is strongly associated with higher observed charges in this dataset.
        - 🎂 Age and ⚖️ BMI show meaningful relationships with insurance charges.
        - 🌳 Nonlinear models can capture interactions that a simple linear baseline may miss.
        - 📍 Region, 🚻 sex, and 👶 number of children show smaller descriptive differences than smoking status in this sample.
        - ⚠️ These are predictive and descriptive associations; they do not establish causation.
        """
    )

    if not importance.empty:
        st.markdown("#### 🌳 Decision Tree feature importance")

        st.dataframe(
            importance.head(15),
            width="stretch",
            hide_index=True,
        )

        if {
            "Feature",
            "importance",
        }.issubset(importance.columns):
            importance["importance"] = pd.to_numeric(
                importance["importance"],
                errors="coerce",
            )

            feature_chart = make_importance_chart(
                importance,
            )

            st.altair_chart(
                feature_chart,
                width="stretch",
            )

        else:
            st.info(
                "ℹ️ Feature-importance data was found, but the expected "
                "'Feature' and 'importance' columns are not both available."
            )

    if not segments.empty:
        st.markdown("#### 🧩 Predicted-cost segmentation")

        st.dataframe(
            segments,
            width="stretch",
            hide_index=True,
        )


# -------------------------------------------------------------------
# Filtered dataset tab
# -------------------------------------------------------------------
with data_tab:
    st.subheader("🗂️ Filtered insurance dataset")

    st.write(
        f"Showing **{len(filtered_data):,}** of **{len(data):,}** "
        "cleaned records based on the 🎛️ sidebar filters."
    )

    if filtered_data.empty:
        st.warning(
            "⚠️ No records match the selected filters. "
            "Broaden one or more sidebar filters."
        )

    else:
        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "👥 Records",
                f"{len(filtered_data):,}",
            )

        with metric2:
            st.metric(
                "💵 Avg. charges",
                f"${filtered_data['charges'].mean():,.0f}",
            )

        with metric3:
            st.metric(
                "🎂 Avg. age",
                f"{filtered_data['age'].mean():.1f}",
            )

        with metric4:
            st.metric(
                "⚖️ Avg. BMI",
                f"{filtered_data['bmi'].mean():.1f}",
            )

        st.dataframe(
            filtered_data,
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            "#### 📋 Descriptive statistics for filtered records"
        )

        st.dataframe(
            filtered_data.describe(
                include="all"
            ).transpose(),
            width="stretch",
        )

        st.download_button(
            "⬇️ Download filtered CSV",
            data=filtered_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="filtered_insurance_data.csv",
            mime="text/csv",
            width="stretch",
        )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()
st.caption(
    "🏥 Medical Insurance Charges Regression Portfolio Project | "
    "Educational ML demonstration"
)
