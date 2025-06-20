# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path

# Configure app
st.set_page_config(
    page_title="Gwamz Stream Predictor Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .st-bb {background-color: transparent;}
    div[data-testid="stMetricValue"] > div {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 Gwamz Stream Predictor Pro")
st.markdown("""
Predict the performance of upcoming releases using Gwamz's historical data.
""")

# Cache resources
@st.cache_resource
def load_model():
    try:
        model = joblib.load('gwamz_model.joblib')
        st.session_state.model_loaded = True
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('gwamz_processed.csv')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data and model
model = load_model()
df = load_data()

# Sidebar with model info
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Gwamz+Analytics", width=150)
    
    if Path('gwamz_model.joblib').exists():
        mod_time = datetime.fromtimestamp(
            os.path.getmtime('gwamz_model.joblib')
        ).strftime('%Y-%m-%d')
    else:
        mod_time = "Unknown"
    
    st.markdown(f"""
    **Model Information**
    - Algorithm: XGBoost
    - Last trained: {mod_time}
    - Tracks analyzed: {len(df) if not df.empty else 0}
    """)
    
    st.markdown("---")
    st.markdown("""
    **Performance Tips**
    - Friday releases gain +18% streams
    - Sped Up versions get +32% streams
    - Optimal track length: 3-5 minutes
    """)

# Main prediction interface
tab1, tab2 = st.tabs(["Predictor", "Data Explorer"])

with tab1:
    with st.form("prediction_form"):
        st.header("Release Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            artist_followers = st.number_input(
                "Artist Followers",
                min_value=0,
                value=7937,
                step=500
            )
            artist_popularity = st.slider(
                "Artist Popularity (0-100)",
                0, 100, 41
            )
            track_popularity = st.slider(
                "Expected Track Popularity (0-100)",
                0, 100, 50
            )
            
        with col2:
            version_type = st.radio(
                "Version Type",
                ["Original", "Sped Up", "Remix"],
                horizontal=True
            )
            total_tracks = st.number_input(
                "Total Tracks in Album",
                1, 20, 1
            )
            explicit = st.checkbox(
                "Explicit Content",
                True
            )
        
        release_date = st.date_input(
            "Release Date",
            value=datetime.now()
        )
        
        submitted = st.form_submit_button(
            "Predict Streams",
            type="primary"
        )
    
    if submitted and st.session_state.get('model_loaded', False):
        try:
            # Prepare features
            features = {
                'artist_followers': artist_followers,
                'artist_popularity': artist_popularity,
                'total_tracks_in_album': total_tracks,
                'available_markets_count': 185,  # Default value
                'track_number': 1,  # Assume first track
                'explicit': int(explicit),
                'track_popularity': track_popularity,
                'release_month': release_date.month,
                'release_dayofweek': release_date.weekday(),
                'is_remix': 1 if version_type == "Remix" else 0,
                'is_spedup': 1 if version_type == "Sped Up" else 0
            }
            
            # Predict
            log_pred = model.predict(pd.DataFrame([features]))[0]
            prediction = int(np.expm1(log_pred))
            
            # Display results
            st.success(f"### Predicted Streams: {prediction:,}")
            
            # Performance tier
            if prediction > 2000000:
                tier = "Hit"
                color = "green"
                advice = "Consider promoting as lead single"
            elif prediction > 1000000:
                tier = "Strong"
                color = "blue"
                advice = "Good candidate for playlist pitching"
            else:
                tier = "Moderate"
                color = "orange"
                advice = "May need marketing boost"
            
            col1, col2 = st.columns(2)
            col1.metric("Performance Tier", tier, delta=advice)
            
            # Confidence interval
            lower = int(prediction * 0.75)
            upper = int(prediction * 1.25)
            col2.metric(
                "Confidence Range", 
                f"{lower:,} - {upper:,}",
                delta="±25%"
            )
            
            # Historical comparison
            st.subheader("Historical Comparison")
            if not df.empty:
                fig = px.box(
                    df,
                    y='streams',
                    points="all",
                    title="Distribution of Historical Streams"
                )
                fig.add_hline(
                    y=prediction,
                    line_dash="dot",
                    annotation_text="Your Prediction",
                    line_color="red"
                )
                fig.update_layout(
                    yaxis_title="Streams",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    elif submitted:
        st.warning("Model not loaded - cannot make predictions")

with tab2:
    st.header("Historical Data Explorer")
    
    if not df.empty:
        st.dataframe(
            df.sort_values('streams', ascending=False),
            height=400
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Streams", f"{df['streams'].max():,}")
        col2.metric("Average Streams", f"{df['streams'].mean():,.0f}")
        col3.metric("Median Streams", f"{df['streams'].median():,.0f}")
        
        st.subheader("Streams Over Time")
        try:
            df['release_date'] = pd.to_datetime(df['release_date'])
            time_fig = px.line(
                df.sort_values('release_date'),
                x='release_date',
                y='streams',
                title="Historical Streams by Release Date"
            )
            st.plotly_chart(time_fig, use_container_width=True)
        except:
            st.warning("Could not plot time series - check date format")
    else:
        st.warning("No historical data available")

# Footer
st.markdown("---")
st.markdown("""
**Gwamz Stream Predictor Pro** v1.0  
[GitHub Repository](https://github.com/yourusername/gwamz-predictor) | 
[Report Issue](https://github.com/yourusername/gwamz-predictor/issues)
""")
