import streamlit as st
import pandas as pd
import pickle
import plotly.express as px


# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="House Price Intelligence AI",
    page_icon="🏠",
    layout="wide"
)


# ==================================
# CUSTOM STYLE
# ==================================

st.markdown("""
<style>

.main{
background:#f8fafc;
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
border-radius:18px;
box-shadow:0px 5px 15px rgba(0,0,0,0.08);

}

.stButton button{

width:100%;
height:45px;
border-radius:10px;
font-size:18px;

}

</style>
""", unsafe_allow_html=True)



# ==================================
# LOAD FILES
# ==================================

@st.cache_data
def load_data():

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


df = load_data()

model = load_model()



# ==================================
# HEADER
# ==================================

st.title(
"🏠 House Price Intelligence Platform"
)


st.markdown(
"""
### AI Powered Real Estate Valuation System

Predict property prices using Machine Learning Regression.
"""
)


st.success(
"Powered by Random Forest Regression Model"
)



# ==================================
# DASHBOARD KPI
# ==================================

st.header(
"📊 Real Estate Overview"
)


a,b,c,d,e = st.columns(5)


a.metric(
"🏠 Properties",
len(df)
)


b.metric(
"💰 Average Price",
f"{df['Price'].mean()/1000000:.1f}M"
)


c.metric(
"📐 Avg Area",
f"{df['Area'].mean():.0f} sq ft"
)


d.metric(
"📍 Locations",
df["Location"].nunique()
)


e.metric(
"🔥 Highest Price",
f"{df['Price'].max()/1000000:.1f}M"
)



# ==================================
# TABS
# ==================================

tab1,tab2,tab3,tab4 = st.tabs(
[
"📈 Market Analytics",
"🤖 AI Prediction",
"🧠 Model Information",
"📥 Data Export"
]
)



# ==================================
# MARKET ANALYTICS
# ==================================

with tab1:


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
            df.groupby("Location")
            ["Price"]
            .mean()
            .reset_index()
        )


        fig2 = px.bar(

            location_price,

            x="Location",

            y="Price",

            title="Average Price By Location"

        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )



    fig3 = px.scatter(

        df,

        x="Area",

        y="Price",

        color="Location",

        size="Bedrooms",

        title="Area vs Price Relationship"

    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )



# ==================================
# AI PREDICTION
# ==================================

with tab2:


    st.header(
    "🏠 Predict Your Property Value"
    )


    col1,col2 = st.columns(2)


    with col1:

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


        floors = st.selectbox(
            "Floors",
            [1,2,3]
        )


    with col2:


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
        "🚀 Predict House Price"
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


        price = prediction[0]


        st.success(

        f"""
        🏠 Estimated Property Value

        ## Rs {price:,.0f}

        """

        )



# ==================================
# MODEL INFORMATION
# ==================================

with tab3:


    st.header(
    "🧠 Machine Learning Details"
    )


    x,y,z = st.columns(3)


    x.metric(
    "Algorithm",
    "Random Forest"
    )


    y.metric(
    "Learning",
    "Supervised"
    )


    z.metric(
    "Task",
    "Regression"
    )


    st.markdown(
"""
### Workflow
