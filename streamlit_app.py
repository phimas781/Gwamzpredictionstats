import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Performance tiers with color coding
PERFORMANCE_TIERS = {
    'Hit': {'min': 2000000, 'color': '#2ecc71'},
    'Strong': {'min': 1000000, 'color': '#3498db'},
    'Moderate': {'min': 0, 'color': '#e67e22'}
}

def get_performance_tier(streams):
    for tier, spec in PERFORMANCE_TIERS.items():
        if streams >= spec['min']:
            return tier, spec['color']

# Enhanced prediction function
def predict_streams(inputs):
    # Feature transformations
    inputs['days_since_first_release'] = (inputs['release_date'] - pd.to_datetime("2021-01-01")).days
    inputs['album_track_ratio'] = inputs['track_number'] / inputs['total_tracks_in_album']
    inputs['followers_per_market'] = inputs['artist_followers'] / inputs['available_markets_count']
    
    # Convert to DataFrame
    pred_df = pd.DataFrame([inputs])
    
    # Ensure feature order matches training
    pred_df = pred_df[features]
    
    # Predict and return
    return np.expm1(model.predict(pred_df)[0])

# Layout configuration
st.set_page_config(layout="wide", page_title="Gwamz Analytics Pro")

# Main app interface
with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Release Parameters")
        with st.form("prediction_form"):
            # Interactive controls...
            
            if st.form_submit_button("Predict Streams"):
                # Process inputs and get prediction
                predicted = predict_streams(user_inputs)
                tier, color = get_performance_tier(predicted)
                
                # Store in session state
                st.session_state.prediction = {
                    'value': predicted,
                    'tier': tier,
                    'color': color
                }

    with col2:
        st.header("Performance Analysis")
        if 'prediction' in st.session_state:
            # Visualize prediction
            fig = px.bar(
                x=[st.session_state.prediction['value']],
                orientation='h',
                title=f"Predicted Streams: {st.session_state.prediction['value']:,.0f}",
                color_discrete_sequence=[st.session_state.prediction['color']]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show historical comparison
            st.plotly_chart(
                px.histogram(
                    df, 
                    x='streams',
                    nbins=20,
                    title="How this compares to historical releases"
                ).add_vline(x=st.session_state.prediction['value']),
                use_container_width=True
            )

# Add strategy recommendations
if 'prediction' in st.session_state:
    st.subheader("Optimization Recommendations")
    
    if st.session_state.prediction['tier'] == 'Hit':
        st.success("""
        🚀 **Maximize this hit potential:**
        - Push for editorial playlists
        - Invest in performance marketing
        - Release companion content
        """)
    elif st.session_state.prediction['tier'] == 'Strong':
        st.info("""
        💎 **Enhance this strong performer:**
        - Target genre-specific playlists  
        - Collaborate with micro-influencers
        - Release alternate versions
        """)
    else:
        st.warning("""
        🔍 **Improvement opportunities:**
        - Test different release timing
        - Consider remix collaborations
        - Boost pre-release promotion
        """)
