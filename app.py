import streamlit as st
import pandas as pd
import pickle
import plotly.express as px


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="House Price AI Intelligence",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
"""
<style>

body {
background-color:#f8fafc;
}

.main-title {
font-size:45px;
font-weight:700;
color:#0f172a;
}

.subtitle {
font-size:20px;
color:#475569;
}

.card {

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 15px rgba(0,0,0,0.08);

}

</style>
""",
unsafe_allow_html=True
)



# ==========================================
# LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "house_prices.csv"
    )


@st.cache_resource
def load_model():

    return pickle.load(
        open(
            "house_price_model.pkl",
            "rb"
        )
    )


df = load_dataset()

model = load_model()



# ==========================================
# HEADER
# ==========================================


st.markdown(
"""
<div class="main-title">

🏠 House Price Intelligence Platform

</div>

<div class="subtitle">

AI Powered Real Estate Valuation System

</div>

""",
unsafe_allow_html=True
)


st.info(
"""
This platform uses Machine Learning to analyze property
features and estimate market value.
"""
)



# ==========================================
# KPI DASHBOARD
# ==========================================


st.header("📊 Executive Overview")


c1,c2,c3,c4 = st.columns(4)


with c1:
    st.metric(
        "Total Houses",
        len(df)
    )


with c2:
    st.metric(
        "Average Price",
        f"{df['Price'].mean()/1000000:.2f} M"
    )


with c3:
    st.metric(
        "Average Area",
        f"{df['Area'].mean():.0f} sqft"
    )


with c4:
    st.metric(
        "Locations",
        df["Location"].nunique()
    )



# ==========================================
# ANALYTICS SECTION
# ==========================================


st.header(
"📈 Real Estate Analytics"
)


tab1,tab2,tab3 = st.tabs(
[
"Price Analysis",
"Location Analysis",
"Property Trends"
]
)



with tab1:


    fig = px.histogram(

        df,

        x="Price",

        title="Price Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



with tab2:


    location_data = (

        df.groupby("Location")
        ["Price"]
        .mean()
        .reset_index()

    )


    fig = px.bar(

        location_data,

        x="Location",

        y="Price",

        title="Average Price by Location"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



with tab3:


    fig = px.scatter(

        df,

        x="Area",

        y="Price",

        color="Location",

        size="Bedrooms",

        title="Area vs Price Relationship"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ==========================================
# AI PREDICTION
# ==========================================


st.header(
"🤖 AI House Price Predictor"
)


left,right = st.columns(2)


with left:


    area = st.slider(

        "Area (sq ft)",

        500,

        10000,

        2500

    )


    bedrooms = st.selectbox(

        "Bedrooms",

        [1,2,3,4,5,6]

    )


    bathrooms = st.selectbox(

        "Bathrooms",

        [1,2,3,4,5]

    )



with right:


    floors = st.selectbox(

        "Floors",

        [1,2,3]

    )


    age = st.slider(

        "Property Age",

        0,

        50,

        5

    )


    location = st.selectbox(

        "Location",

        df["Location"].unique()

    )




if st.button(
"🚀 Predict Property Price"
):


    input_data = pd.DataFrame({

        "Area":[area],

        "Bedrooms":[bedrooms],

        "Bathrooms":[bathrooms],

        "Floors":[floors],

        "Age":[age],

        "Location":[location]

    })


    prediction = model.predict(
        input_data
    )


    st.success(

        f"""
        🏠 Estimated Property Value

        ## Rs {prediction[0]:,.0f}

        """

    )



# ==========================================
# MACHINE LEARNING INFORMATION
# ==========================================


st.header(
"🧠 Machine Learning Methodology"
)


a,b,c = st.columns(3)


a.metric(
"Algorithm",
"Random Forest"
)


b.metric(
"Learning Type",
"Supervised"
)


c.metric(
"Problem",
"Regression"
)



st.markdown(
"""
### Workflow
