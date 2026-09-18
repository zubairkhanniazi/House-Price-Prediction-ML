"""
House Price Intelligence Platform
----------------------------------
A Streamlit dashboard for exploring real-estate data and generating
AI-powered property price estimates using a pre-trained regression model.
"""

import pandas as pd
import pickle
import plotly.express as px
import streamlit as st

DATA_PATH = "house_prices.csv"
MODEL_PATH = "house_price_model.pkl"
REQUIRED_COLUMNS = ["Area", "Bedrooms", "Bathrooms", "Floors", "Age", "Location", "Price"]


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price AI Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# STYLING
# ==========================================

def inject_custom_css() -> None:
    """Apply the platform's visual theme."""
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 18px;
            color: #475569;
            margin-top: 4px;
        }
        .section-divider {
            margin: 1.5rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }
        .footer-note {
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# DATA & MODEL LOADING
# ==========================================

@st.cache_data(show_spinner="Loading property dataset...")
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of required columns missing from the dataset, if any."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


# ==========================================
# UI SECTIONS
# ==========================================

def render_header() -> None:
    st.markdown(
        """
        <div class="main-title">🏠 House Price Intelligence Platform</div>
        <div class="subtitle">AI-Powered Real Estate Valuation System</div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This platform uses Machine Learning to analyze property features "
        "and estimate market value in real time."
    )


def render_kpi_dashboard(df: pd.DataFrame) -> None:
    st.header("📊 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Listings", f"{len(df):,}")
    col2.metric("Average Price", f"Rs {df['Price'].mean() / 1_000_000:.2f} M")
    col3.metric("Average Area", f"{df['Area'].mean():.0f} sqft")
    col4.metric("Locations Covered", df["Location"].nunique())


def render_analytics(df: pd.DataFrame) -> None:
    st.header("📈 Real Estate Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Distribution", "Location Analysis", "Area vs Price", "Correlations"]
    )

    with tab1:
        fig = px.histogram(df, x="Price", nbins=40, title="Price Distribution")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        location_data = (
            df.groupby("Location")["Price"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            location_data,
            x="Location",
            y="Price",
            title="Average Price by Location",
            color="Price",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(
            df,
            x="Area",
            y="Price",
            color="Location",
            size="Bedrooms",
            title="Area vs Price Relationship",
            hover_data=["Bedrooms", "Bathrooms", "Age"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Correlation values range from -1 to 1. Values close to 1 indicate "
            "a strong positive relationship with Price; values close to -1 "
            "indicate a strong negative relationship."
        )


def render_predictor(df: pd.DataFrame, model) -> None:
    st.header("🤖 AI House Price Predictor")

    left, right = st.columns(2)

    with left:
        area = st.slider("Area (sq ft)", 500, 10000, 2500, step=50)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])

    with right:
        floors = st.selectbox("Floors", [1, 2, 3])
        age = st.slider("Property Age (years)", 0, 50, 5)
        location = st.selectbox("Location", sorted(df["Location"].unique()))

    predict_clicked = st.button("🚀 Predict Property Price", type="primary")

    if predict_clicked:
        input_data = pd.DataFrame(
            {
                "Area": [area],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Floors": [floors],
                "Age": [age],
                "Location": [location],
            }
        )

        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        price_per_sqft = prediction / area if area else 0

        st.success(f"🏠 Estimated Property Value: **Rs {prediction:,.0f}**")

        m1, m2 = st.columns(2)
        m1.metric("Price per sqft", f"Rs {price_per_sqft:,.0f}")
        m2.metric(
            "Comparable Avg. (same location)",
            f"Rs {df.loc[df['Location'] == location, 'Price'].mean():,.0f}",
        )

        _log_prediction(input_data.assign(Predicted_Price=prediction))


def _log_prediction(record: pd.DataFrame) -> None:
    """Keep a running, in-session history of predictions for review/export."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame()

    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, record], ignore_index=True
    )


def render_prediction_history() -> None:
    history = st.session_state.get("prediction_history")

    if history is None or history.empty:
        return

    st.header("🕘 Prediction History")
    st.dataframe(history, use_container_width=True)

    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def render_model_insights(model) -> None:
    st.header("🧠 Machine Learning Methodology")

    a, b, c = st.columns(3)
    a.metric("Algorithm", "Random Forest")
    b.metric("Learning Type", "Supervised")
    c.metric("Problem", "Regression")

    st.markdown(
        """
        ### Workflow
        1. **Data Collection** — Historical property records were gathered, including area,
           bedrooms, bathrooms, floors, age, and location.
        2. **Data Cleaning** — Missing values and outliers were handled to ensure data quality.
        3. **Feature Engineering** — Categorical variables (e.g. Location) were encoded, and
           numerical features were scaled where needed.
        4. **Model Training** — A Random Forest Regressor was trained on the processed dataset
           to learn the relationship between property features and price.
        5. **Model Evaluation** — Performance was validated using standard regression metrics
           (e.g. RMSE, R²) on a held-out test set.
        6. **Deployment** — The trained model was integrated into this Streamlit application
           for real-time price predictions.
        """
    )

    _render_feature_importance(model)


def _render_feature_importance(model) -> None:
    """Show feature importance if the underlying model supports it."""
    try:
        importances = model.named_steps["regressor"].feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    except (AttributeError, KeyError):
        # Model isn't a pipeline, or doesn't expose feature importances — skip silently.
        return

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_footer() -> None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footer-note'>House Price Intelligence Platform · "
        "Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR
# ==========================================

def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.title("🏠 Navigation")
        st.markdown(
            "Use the sections on the main page to explore analytics, "
            "run predictions, and review model details."
        )
        st.markdown("---")
        st.subheader("Dataset Snapshot")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Locations:** {df['Location'].nunique()}")

        if st.checkbox("Show raw data"):
            st.dataframe(df, use_container_width=True)


# ==========================================
# MAIN APP
# ==========================================

def main() -> None:
    inject_custom_css()

    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error(f"Dataset not found at '{DATA_PATH}'. Please check the file path.")
        st.stop()

    missing_cols = validate_dataset(df)
    if missing_cols:
        st.error(f"Dataset is missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please check the file path.")
        st.stop()

    render_sidebar(df)
    render_header()
    render_kpi_dashboard(df)
    render_analytics(df)
    render_predictor(df, model)
    render_prediction_history()
    render_model_insights(model)
    render_footer()


if __name__ == "__main__":
    main()
