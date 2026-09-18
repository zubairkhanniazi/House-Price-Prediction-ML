import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="House Price Intelligence Platform",
    page_icon="🏠",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
"""
<style>

.main{
background-color:#f6f8fc;
}


h1{
color:#0f172a;
}


h2{
color:#1e293b;
}


div[data-testid="metric-container"]{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);

}

</style>

""",
unsafe_allow_html=True
)



# =====================================
# LOAD DATA
# =====================================


@st.cache_data
def load_data():

    return pd.read_csv(
        "house_prices.csv"
    )


df = load_data()



# =====================================
# LOAD MODEL
# =====================================


@st.cache_resource
def load_model():

    model = pickle.load(
        open(
            "house_price_model.pkl",
            "rb"
        )
    )

    return model



model = load_model()



# =====================================
# HEADER
# =====================================


st.title(
"🏠 House Price Intelligence Platform"
)


st.markdown(
"""
### AI Powered Real Estate Valuation System

Predict property prices using Machine Learning Regression Models.
"""
)


st.info(
"""
This platform analyzes housing features and predicts
estimated market prices using Artificial Intelligence.
"""
)



# =====================================
# SIDEBAR
# =====================================


st.sidebar.header(
"🏠 Property Information"
)


area = st.sidebar.number_input(
"Area (sq ft)",
500,
10000,
2500
)


bedrooms = st.sidebar.number_input(
"Bedrooms",
1,
10,
3
)


bathrooms = st.sidebar.number_input(
"Bathrooms",
1,
10,
2
)


floors = st.sidebar.number_input(
"Floors",
1,
5,
2
)


age = st.sidebar.number_input(
"House Age",
0,
100,
10
)


location = st.sidebar.selectbox(
"Location",
df["Location"].unique()
)



# =====================================
# EXECUTIVE OVERVIEW
# =====================================


st.header(
"📊 Real Estate Overview"
)


c1,c2,c3,c4 = st.columns(4)


with c1:

    st.metric(
    "Total Properties",
    len(df)
    )


with c2:

    st.metric(
    "Average Price",
    round(
    df["Price"].mean()
    )
    )


with c3:

    st.metric(
    "Average Area",
    round(
    df["Area"].mean()
    )
    )


with c4:

    st.metric(
    "Locations",
    df["Location"].nunique()
    )



# =====================================
# MARKET ANALYSIS
# =====================================


st.header(
"📈 Market Analysis"
)


col1,col2 = st.columns(2)



with col1:

    fig1 = px.histogram(

        df,

        x="Price",

        title="House Price Distribution"

    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )



with col2:


    location_price = (
        df.groupby("Location")["Price"]
        .mean()
        .reset_index()
    )


    fig2 = px.bar(

        location_price,

        x="Location",

        y="Price",

        title="Average Price by Location"

    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )



# =====================================
# AREA VS PRICE
# =====================================


st.header(
"🏘 Property Relationship Analysis"
)


fig3 = px.scatter(

df,

x="Area",

y="Price",

color="Location",

title="Area vs Property Price"

)


st.plotly_chart(
fig3,
use_container_width=True
)



# =====================================
# AI PREDICTION
# =====================================


st.header(
"🤖 AI House Price Prediction"
)



if st.button(
"Predict House Price"
):


    input_data = pd.DataFrame(

    {

    "Area":[area],

    "Bedrooms":[bedrooms],

    "Bathrooms":[bathrooms],

    "Floors":[floors],

    "Age":[age],

    "Location":[location]

    }

    )


    prediction = model.predict(
        input_data
    )


    st.success(

    f"""
    🏠 Estimated House Price

    Rs {prediction[0]:,.0f}

    """

    )



# =====================================
# MODEL INFORMATION
# =====================================


st.header(
"🧠 Machine Learning Information"
)



col1,col2,col3 = st.columns(3)



col1.metric(
"Problem Type",
"Regression"
)


col2.metric(
"Algorithm",
"Random Forest"
)


col3.metric(
"Prediction",
"House Price"
)



st.markdown(
"""
### ML Workflow

"""
)



# =====================================
# DOWNLOAD DATA
# =====================================


st.header(
"📥 Download Dataset"
)


csv = df.to_csv(
index=False
)


st.download_button(

"Download Housing Data",

csv,

"house_prices.csv",

"text/csv"

)



# =====================================
# FOOTER
# =====================================


st.markdown(
"""
---
🚀 Built using Python | Machine Learning | Regression | Streamlit
"""
)
