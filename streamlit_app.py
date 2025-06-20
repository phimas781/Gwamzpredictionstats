import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model
model = joblib.load('gwamz_stream_predictor.joblib')

# App UI
st.title('🎵 Gwamz Stream Predictor')

with st.form("prediction_form"):
    st.header("Release Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        artist_followers = st.number_input("Artist Followers", value=7937)
        artist_popularity = st.slider("Artist Popularity (0-100)", 0, 100, 45)
        track_popularity = st.slider("Track Popularity (0-100)", 0, 100, 50)
        
    with col2:
        version_type = st.radio("Version Type", ["Original", "Sped Up", "Remix"])
        explicit = st.checkbox("Explicit Content", True)
        release_date = st.date_input("Release Date")
    
    if st.form_submit_button("Predict Streams"):
        # Prepare features
        features = {
            'artist_followers': artist_followers,
            'artist_popularity': artist_popularity,
            'total_tracks_in_album': 1,  # Assume single
            'available_markets_count': 185,  # Default
            'track_number': 1,  # Assume lead track
            'explicit': int(explicit),
            'track_popularity': track_popularity,
            'release_year': release_date.year,
            'release_month': release_date.month,
            'release_day': release_date.day,
            'days_since_first_release': (release_date - pd.to_datetime("2021-01-01").date()).days,
            'is_remix': 1 if version_type == "Remix" else 0,
            'is_spedup': 1 if version_type == "Sped Up" else 0,
            'is_instrumental': 0
        }
        
        # Predict
        log_pred = model.predict(pd.DataFrame([features]))[0]
        prediction = int(np.expm1(log_pred))
        
        # Show result
        st.success(f"Predicted Streams: **{prediction:,}**")
