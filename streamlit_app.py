import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# Load model
try:
    model = joblib.load('gwamz_optimized_model.joblib')
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Define features exactly as used in training
features = [
    'artist_followers', 'artist_popularity', 'total_tracks_in_album',
    'available_markets_count', 'track_number', 'explicit', 'track_popularity',
    'release_month', 'days_since_first_release', 'is_remix', 'is_spedup',
    'growth_since_last', 'stream_momentum', 'album_track_ratio',
    'followers_per_market'
]

def predict_streams(inputs):
    """Make prediction with proper feature formatting"""
    # Convert to DataFrame with correct column order
    pred_df = pd.DataFrame([inputs])[features] 
    
    # Handle missing values (if any)
    pred_df = pred_df.fillna({
        'growth_since_last': 0,
        'stream_momentum': pred_df['artist_followers'].mean() / 1000
    })
    
    return np.expm1(model.predict(pred_df)[0])

# App layout
st.set_page_config(layout="wide")

with st.form("prediction_form"):
    st.header("Release Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        artist_followers = st.number_input("Artist Followers", value=7937)
        artist_popularity = st.slider("Artist Popularity (0-100)", 0, 100, 45)
        track_popularity = st.slider("Track Popularity (0-100)", 0, 100, 50)
        
    with col2:
        version_type = st.radio("Version Type", ["Original", "Sped Up", "Remix"])
        explicit = st.checkbox("Explicit Content", True)
        release_date = st.date_input("Release Date", datetime.now())
    
    submit_button = st.form_submit_button("Predict Streams")

if submit_button:
    # Prepare input dictionary
    user_inputs = {
        'artist_followers': artist_followers,
        'artist_popularity': artist_popularity,
        'total_tracks_in_album': 1,  # Assume single
        'available_markets_count': 185,
        'track_number': 1,
        'explicit': int(explicit),
        'track_popularity': track_popularity,
        'release_month': release_date.month,
        'days_since_first_release': (release_date - pd.to_datetime("2021-01-01").date()).days,
        'is_remix': 1 if version_type == "Remix" else 0,
        'is_spedup': 1 if version_type == "Sped Up" else 0,
        'growth_since_last': 0.05,  # Default growth rate
        'stream_momentum': artist_followers / 1000,  # Estimated
        'album_track_ratio': 1.0,  # Single track
        'followers_per_market': artist_followers / 185
    }
    
    try:
        predicted = predict_streams(user_inputs)
        st.success(f"Predicted Streams: {predicted:,.0f}")
        
        # Show feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            fig = px.bar(
                x=model.feature_importances_,
                y=features,
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
