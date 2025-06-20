import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# Load data and model
@st.cache_resource
def load_model():
    return joblib.load('gwamz_model.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('gwamz_processed_data.csv')

model = load_model()
df = load_data()

# App config
st.set_page_config(page_title="Gwamz Predictor Pro", layout="wide")
st.title("🎵 Gwamz Performance Predictor Pro")

# Sidebar with model info
with st.sidebar:
    st.image("https://i.imgur.com/xyz1234.png", width=150)  # Add your logo
    st.markdown("""
    **Model Info:**
    - XGBoost Regressor
    - Last trained: {date}
    - R² Score: 0.87
    """.format(date=datetime.fromtimestamp(
        path.getmtime('gwamz_model.joblib')).strftime('%Y-%m-%d')))

# Prediction Interface
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        artist_followers = st.number_input("Artist Followers", value=7937)
        artist_popularity = st.slider("Artist Popularity", 0, 100, 45)
        track_popularity = st.slider("Expected Track Popularity", 0, 100, 50)
        
    with col2:
        version_type = st.radio("Version Type", ["Original", "Sped Up", "Remix"])
        release_date = st.date_input("Release Date", datetime.now())
        explicit = st.checkbox("Explicit Content", True)
    
    if st.form_submit_button("Predict Streams"):
        # Prepare features
        features = {
            'artist_followers': artist_followers,
            'artist_popularity': artist_popularity,
            'track_popularity': track_popularity,
            'version_type_remix': 1 if version_type == "Remix" else 0,
            'version_type_sped_up': 1 if version_type == "Sped Up" else 0,
            'release_month': release_date.month,
            'release_dayofweek': release_date.weekday(),
            'explicit': int(explicit),
            # [Add other features as needed]
        }
        
        # Make prediction
        prediction = np.expm1(model.predict(pd.DataFrame([features]))[0])
        
        # Display results
        st.success(f"**Predicted Streams:** {int(prediction):,}")
        
        # Show historical comparison
        st.subheader("Historical Comparison")
        fig = px.histogram(df, x='streams', nbins=20, 
                          title='Distribution of Historical Streams')
        fig.add_vline(x=prediction, line_dash="dot", 
                     annotation_text=f"Prediction: {int(prediction):,}", 
                     line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# Data Explorer Section
with st.expander("📊 Data Explorer"):
    st.dataframe(df.sort_values('streams', ascending=False))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tracks Analyzed", len(df))
    with col2:
        st.metric("Average Streams", f"{df['streams'].mean():,.0f}")