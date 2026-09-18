"""
House Price Intelligence Platform
----------------------------------
A Streamlit dashboard for exploring real-estate data and generating
AI-powered property price estimates using a pre-trained regression model.
"""

import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "house_prices.csv"
MODEL_PATH = "house_price_model.pkl"
REQUIRED_COLUMNS = ["Area", "Bedrooms", "Bathrooms", "Floors", "Age", "Location", "Price"]

ACCENT = "#6366f1"
ACCENT_DARK = "#4338ca"


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
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .hero {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        }}
        .hero-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .hero-subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            border: 1px solid #eef0f5;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
        }}

        section[data-testid="stSidebar"] {{
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}

        .section-divider {{
            margin: 1.8rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }}
        .footer-note {{
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            padding-top: 1.5rem;
        }}
        .badge {{
            display: inline-block;
            background: #eef2ff;
            color: {ACCENT_DARK};
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            margin-right: 6px;
        }}
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
# HERO / HEADER
# ==========================================

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🏠 House Price Intelligence Platform</div>
            <div class="hero-subtitle">AI-powered real estate valuation, market analytics
            and portfolio insight — all in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR — FILTERS
# ==========================================

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataframe used by
    the KPI dashboard and analytics charts."""
    with st.sidebar:
        st.title("🏠 Control Panel")
        st.caption("Filter the market data explored below. "
                    "The predictor further down always uses the full dataset.")

        st.markdown("---")
        st.subheader("Filters")

        locations = sorted(df["Location"].unique())
        selected_locations = st.multiselect(
            "Location", locations, default=locations
        )

        price_min, price_max = int(df["Price"].min()), int(df["Price"].max())
        price_range = st.slider(
            "Price Range (Rs)", price_min, price_max, (price_min, price_max)
        )

        area_min, area_max = int(df["Area"].min()), int(df["Area"].max())
        area_range = st.slider(
            "Area Range (sqft)", area_min, area_max, (area_min, area_max)
        )

        filtered = df[
            df["Location"].isin(selected_locations)
            & df["Price"].between(*price_range)
            & df["Area"].between(*area_range)
        ]

        st.markdown("---")
        st.subheader("Dataset Snapshot")
        st.write(f"**Showing:** {len(filtered):,} / {len(df):,} listings")
        st.write(f"**Columns:** {df.shape[1]}")

        if st.checkbox("Show raw data"):
            st.dataframe(filtered, use_container_width=True)

        if filtered.empty:
            st.warning("No listings match the current filters.")

    return filtered


# ==========================================
# KPI DASHBOARD
# ==========================================

def render_kpi_dashboard(df: pd.DataFrame) -> None:
    st.header("📊 Executive Overview")

    if df.empty:
        st.info("Adjust the filters in the sidebar to see results.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Listings", f"{len(df):,}")
    col2.metric("Average Price", f"Rs {df['Price'].mean() / 1_000_000:.2f} M")
    col3.metric("Average Area", f"{df['Area'].mean():.0f} sqft")
    col4.metric("Locations Covered", df["Location"].nunique())


# ==========================================
# MARKET HIGHLIGHTS
# ==========================================

def render_market_highlights(df: pd.DataFrame) -> None:
    if df.empty:
        return

    st.header("⭐ Market Highlights")

    working = df.copy()
    working["Price per sqft"] = (working["Price"] / working["Area"]).round(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="badge">TOP</span> Most Expensive Listings', unsafe_allow_html=True)
        top_expensive = (
            working.sort_values("Price", ascending=False)
            .head(5)[["Location", "Area", "Bedrooms", "Price"]]
            .reset_index(drop=True)
        )
        st.dataframe(top_expensive, use_container_width=True)

    with col2:
        st.markdown('<span class="badge">VALUE</span> Best Price-per-sqft', unsafe_allow_html=True)
        best_value = (
            working.sort_values("Price per sqft", ascending=True)
            .head(5)[["Location", "Area", "Price", "Price per sqft"]]
            .reset_index(drop=True)
        )
        st.dataframe(best_value, use_container_width=True)


# ==========================================
# ANALYTICS
# ==========================================

def render_analytics(df: pd.DataFrame) -> None:
    st.header("📈 Real Estate Analytics")

    if df.empty:
        st.info("Adjust the filters in the sidebar to see charts.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Distribution", "Location Analysis", "Area vs Price", "Correlations"]
    )

    with tab1:
        fig = px.histogram(df, x="Price", nbins=40, title="Price Distribution",
                            color_discrete_sequence=[ACCENT])
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
            location_data, x="Location", y="Price",
            title="Average Price by Location",
            color="Price", color_continuous_scale="Purples",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(
            df, x="Area", y="Price", color="Location", size="Bedrooms",
            title="Area vs Price Relationship",
            hover_data=["Bedrooms", "Bathrooms", "Age"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix", aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Correlation values range from -1 to 1. Values close to 1 indicate "
            "a strong positive relationship with Price; values close to -1 "
            "indicate a strong negative relationship."
        )


# ==========================================
# PREDICTOR
# ==========================================

def render_predictor(full_df: pd.DataFrame, model) -> None:
    st.header("🤖 AI House Price Predictor")

    left, right = st.columns(2)

    with left:
        area = st.slider("Area (sq ft)", 500, 10000, 2500, step=50)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])

    with right:
        floors = st.selectbox("Floors", [1, 2, 3])
        age = st.slider("Property Age (years)", 0, 50, 5)
        location = st.selectbox("Location", sorted(full_df["Location"].unique()))

    predict_clicked = st.button("🚀 Predict Property Price", type="primary")

    if predict_clicked:
        prediction = _predict_price(
            model, area, bedrooms, bathrooms, floors, age, location
        )
        if prediction is None:
            return

        _render_prediction_result(full_df, prediction, area, location)
        _log_prediction(pd.DataFrame({
            "Area": [area], "Bedrooms": [bedrooms], "Bathrooms": [bathrooms],
            "Floors": [floors], "Age": [age], "Location": [location],
            "Predicted_Price": [prediction],
        }))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    render_comparison_tool(full_df, model)


def _predict_price(model, area, bedrooms, bathrooms, floors, age, location):
    input_data = pd.DataFrame({
        "Area": [area], "Bedrooms": [bedrooms], "Bathrooms": [bathrooms],
        "Floors": [floors], "Age": [age], "Location": [location],
    })
    try:
        return model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def _render_prediction_result(df: pd.DataFrame, prediction: float, area: float, location: str) -> None:
    price_per_sqft = prediction / area if area else 0
    comparable_avg = df.loc[df["Location"] == location, "Price"].mean()
    percentile = (df["Price"] < prediction).mean() * 100

    st.success(f"🏠 Estimated Property Value: **Rs {prediction:,.0f}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Price per sqft", f"Rs {price_per_sqft:,.0f}")
    m2.metric("Comparable Avg. (same location)", f"Rs {comparable_avg:,.0f}")
    m3.metric("Market Percentile", f"{percentile:.0f}th")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentile,
        title={"text": "Where this price ranks in the overall market"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": ACCENT},
            "steps": [
                {"range": [0, 33], "color": "#e0e7ff"},
                {"range": [33, 66], "color": "#c7d2fe"},
                {"range": [66, 100], "color": "#a5b4fc"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_tool(full_df: pd.DataFrame, model) -> None:
    st.subheader("⚖️ Compare Two Properties")

    col_a, col_b = st.columns(2)
    configs = {}

    for label, col in [("Property A", col_a), ("Property B", col_b)]:
        with col:
            st.markdown(f"**{label}**")
            a = st.slider(f"Area (sqft) — {label}", 500, 10000, 2500, step=50, key=f"area_{label}")
            bd = st.selectbox(f"Bedrooms — {label}", [1, 2, 3, 4, 5, 6], key=f"bed_{label}")
            ba = st.selectbox(f"Bathrooms — {label}", [1, 2, 3, 4, 5], key=f"bath_{label}")
            fl = st.selectbox(f"Floors — {label}", [1, 2, 3], key=f"floor_{label}")
            ag = st.slider(f"Age — {label}", 0, 50, 5, key=f"age_{label}")
            loc = st.selectbox(f"Location — {label}", sorted(full_df["Location"].unique()), key=f"loc_{label}")
            configs[label] = (a, bd, ba, fl, ag, loc)

    if st.button("Compare Properties"):
        results = {}
        for label, (a, bd, ba, fl, ag, loc) in configs.items():
            results[label] = _predict_price(model, a, bd, ba, fl, ag, loc)

        if None in results.values():
            return

        diff = results["Property A"] - results["Property B"]
        winner = "Property A" if diff > 0 else "Property B"

        r1, r2, r3 = st.columns(3)
        r1.metric("Property A", f"Rs {results['Property A']:,.0f}")
        r2.metric("Property B", f"Rs {results['Property B']:,.0f}")
        r3.metric("Price Difference", f"Rs {abs(diff):,.0f}",
                   delta=f"{winner} is higher")


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


# ==========================================
# MODEL INSIGHTS
# ==========================================

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
        return

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
    )

    fig = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importance", color_discrete_sequence=[ACCENT],
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

    filtered_df = render_sidebar(df)

    render_hero()
    render_kpi_dashboard(filtered_df)
    render_market_highlights(filtered_df)
    render_analytics(filtered_df)
    render_predictor(df, model)
    render_prediction_history()
    render_model_insights(model)
    render_footer()


if __name__ == "__main__":
    main()
