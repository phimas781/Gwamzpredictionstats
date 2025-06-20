# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# App Configuration
st.set_page_config(
    page_title="Gwamz Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .performance-hit { color: #2ecc71; }
    .performance-strong { color: #3498db; }
    .performance-moderate { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

# Cache Resources
@st.cache_resource(ttl=3600)
def load_model():
    try:
        model_path = Path('models/gwamz_model_v2.joblib')
        if not model_path.exists():
            st.error("Model file not found")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    try:
        data_path = Path('data/gwamz_processed_v2.csv')
        if not data_path.exists():
            st.error("Data file not found")
            return pd.DataFrame()
        return pd.read_csv(data_path, parse_dates=['release_date'])
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# App Header
st.title("🎤 Gwamz Advanced Stream Predictor")
st.markdown("""
Predict and optimize your next release's performance using machine learning.
""")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Gwamz+Pro", width=150)
    st.markdown("### Model Information")
    
    if Path('models/gwamz_model_v2.joblib').exists():
        mod_time = datetime.fromtimestamp(
            os.path.getmtime('models/gwamz_model_v2.joblib')
        ).strftime('%Y-%m-%d')
    else:
        mod_time = "Not available"
    
    st.markdown(f"""
    - **Version**: 2.1
    - **Last Trained**: {mod_time}
    - **Tracks Analyzed**: {len(st.session_state.df) if not st.session_state.df.empty else 0}
    """)
    
    st.markdown("---")
    st.markdown("### Performance Tips")
    st.markdown("""
    - 🚀 Friday releases: +18% streams
    - ⚡ Sped Up versions: +32% streams
    - 🎵 3-5 track albums perform best
    - 🔥 Summer releases gain +22% streams
    """)

# Main App Tabs
tab1, tab2, tab3 = st.tabs(["Predictor", "Analytics", "Optimizer"])

with tab1:
    st.header("Stream Prediction Engine")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Artist Metrics")
            artist_followers = st.number_input(
                "Current Followers",
                min_value=0,
                value=7937,
                step=500,
                help="Current number of artist followers on platforms"
            )
            artist_popularity = st.slider(
                "Artist Popularity Score (0-100)",
                0, 100, 45,
                help="Platform-specific popularity metric"
            )
            
            st.subheader("Release Strategy")
            release_date = st.date_input(
                "Planned Release Date",
                min_value=datetime.now(),
                value=datetime.now() + timedelta(days=14)
            )
            total_tracks = st.slider(
                "Total Tracks in Album/EP",
                1, 20, 3
            )
            
        with col2:
            st.subheader("Track Details")
            track_popularity = st.slider(
                "Expected Track Popularity (0-100)",
                0, 100, 65,
                help="Estimated based on track quality and promotion"
            )
            version_type = st.radio(
                "Version Type",
                ["Original", "Sped Up", "Remix"],
                horizontal=True
            )
            explicit = st.toggle(
                "Contains Explicit Content",
                True
            )
            
            st.subheader("Promotion")
            promo_budget = st.select_slider(
                "Marketing Budget",
                options=["None", "Small", "Medium", "Large", "Massive"],
                value="Medium"
            )
            collab_artist = st.selectbox(
                "Featured Artist",
                ["None", "A-List", "B-List", "C-List", "New Artist"]
            )
        
        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True
        )
    
    if submitted and st.session_state.model is not None:
        try:
            # Feature preparation
            features = {
                'artist_followers': artist_followers,
                'artist_popularity': artist_popularity,
                'total_tracks_in_album': total_tracks,
                'available_markets_count': 185,  # Default
                'track_number': 1,  # Assume lead track
                'explicit': int(explicit),
                'track_popularity': track_popularity,
                'release_month': release_date.month,
                'release_dayofweek': release_date.weekday(),
                'days_since_first': (release_date - st.session_state.df['release_date'].min().date()).days,
                'is_remix': 1 if version_type == "Remix" else 0,
                'is_spedup': 1 if version_type == "Sped Up" else 0,
                'artist_growth': 0.05,  # Default growth rate
                'followers_per_market': artist_followers / 185
            }
            
            # Apply business rules
            promo_boost = {
                "None": 1.0,
                "Small": 1.15,
                "Medium": 1.3,
                "Large": 1.5,
                "Massive": 2.0
            }[promo_budget]
            
            collab_boost = {
                "None": 1.0,
                "A-List": 1.4,
                "B-List": 1.25,
                "C-List": 1.1,
                "New Artist": 1.05
            }[collab_artist]
            
            # Make prediction
            input_df = pd.DataFrame([features])
            log_pred = st.session_state.model.predict(input_df)[0]
            base_prediction = int(np.expm1(log_pred))
            adjusted_prediction = int(base_prediction * promo_boost * collab_boost)
            
            # Display results
            st.success("## Prediction Results")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Base Prediction", f"{base_prediction:,}")
            col2.metric("With Promotion", f"{int(base_prediction * promo_boost):,}", 
                       delta=f"+{(promo_boost-1)*100:.0f}%")
            col3.metric("With Collab", f"{adjusted_prediction:,}", 
                       delta=f"+{(collab_boost-1)*100:.0f}%")
            
            # Performance tier
            performance_class = ""
            if adjusted_prediction > 2000000:
                tier = "Hit"
                color = "performance-hit"
                advice = "Potential viral hit - maximize promotion"
            elif adjusted_prediction > 1000000:
                tier = "Strong"
                color = "performance-strong"
                advice = "High performer - focus on playlist pitching"
            else:
                tier = "Moderate"
                color = "performance-moderate"
                advice = "Consider strategic optimizations"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-top:0;">Performance Tier: <span class="{color}">{tier}</span></h3>
                <p>{advice}</p>
                <p><strong>Confidence Range:</strong> {int(adjusted_prediction*0.75):,} - {int(adjusted_prediction*1.25):,} streams (±25%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization
            st.subheader("Performance Simulation")
            
            fig = go.Figure()
            
            # Add prediction
            fig.add_trace(go.Indicator(
                mode="number",
                value=adjusted_prediction,
                number={"prefix": "", "font": {"size": 40}},
                title={"text": "Predicted Streams"},
                domain={'row':0, 'column':0}
            ))
            
            # Add components
            components = {
                "Base": base_prediction,
                "Promotion": int(base_prediction * (promo_boost-1)),
                "Collaboration": int(base_prediction * (collab_boost-1))
            }
            
            fig.add_trace(go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                marker_color=['#3498db', '#2ecc71', '#9b59b6'],
                name="Stream Components"
            ))
            
            fig.update_layout(
                barmode='stack',
                title="Stream Prediction Breakdown",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical comparison
            if not st.session_state.df.empty:
                st.subheader("Historical Context")
                
                hist_fig = px.histogram(
                    st.session_state.df,
                    x='streams',
                    nbins=20,
                    title='Distribution of Historical Streams'
                )
                
                hist_fig.add_vline(
                    x=adjusted_prediction,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Your Prediction"
                )
                
                st.plotly_chart(hist_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.header("Advanced Analytics")
    
    if not st.session_state.df.empty:
        # Time series analysis
        st.subheader("Streams Over Time")
        
        time_fig = px.line(
            st.session_state.df.sort_values('release_date'),
            x='release_date',
            y='streams',
            hover_data=['track_name'],
            title="Historical Performance Trend"
        )
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Version comparison
        st.subheader("Version Performance")
        
        version_data = st.session_state.df.copy()
        version_data['version_type'] = version_data.apply(
            lambda x: 'Sped Up' if x['is_spedup'] else ('Remix' if x['is_remix'] else 'Original'),
            axis=1
        )
        
        version_stats = version_data.groupby('version_type').agg({
            'streams': ['mean', 'count'],
            'track_popularity': 'mean'
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.bar(
                    version_stats,
                    x=version_stats.index,
                    y=('streams', 'mean'),
                    title="Average Streams by Version"
                ),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                px.pie(
                    version_data,
                    names='version_type',
                    title="Version Distribution"
                ),
                use_container_width=True
            )
        
        # Feature importance
        st.subheader("Model Insights")
        
        if hasattr(st.session_state.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.plotly_chart(
                px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top Predictive Factors"
                ),
                use_container_width=True
            )

with tab3:
    st.header("Release Optimizer")
    
    if st.session_state.model is not None:
        with st.form("optimizer_form"):
            st.subheader("Base Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                base_followers = st.number_input(
                    "Current Followers",
                    min_value=0,
                    value=7937,
                    step=500
                )
                base_popularity = st.slider(
                    "Artist Popularity",
                    0, 100, 45
                )
                
            with col2:
                base_track_pop = st.slider(
                    "Track Quality",
                    0, 100, 65,
                    help="Estimated inherent track appeal"
                )
                base_explicit = st.toggle(
                    "Explicit Content",
                    True
                )
            
            st.subheader("Optimization Levers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_version = st.radio(
                    "Test Version Type",
                    ["Original", "Sped Up", "Remix"],
                    index=1
                )
                test_day = st.selectbox(
                    "Test Release Day",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=4
                )
                
            with col2:
                test_promo = st.select_slider(
                    "Test Promotion Level",
                    options=["None", "Small", "Medium", "Large", "Massive"],
                    value="Medium"
                )
                test_collab = st.selectbox(
                    "Test Collaboration",
                    ["None", "A-List", "B-List", "C-List", "New Artist"],
                    index=1
                )
            
            optimize = st.form_submit_button(
                "Run Optimization",
                type="primary"
            )
        
        if optimize:
            try:
                # Base scenario
                base_features = {
                    'artist_followers': base_followers,
                    'artist_popularity': base_popularity,
                    'total_tracks_in_album': 3,
                    'available_markets_count': 185,
                    'track_number': 1,
                    'explicit': int(base_explicit),
                    'track_popularity': base_track_pop,
                    'release_month': datetime.now().month,
                    'release_dayofweek': ["Monday","Tuesday","Wednesday","Thursday",
                                        "Friday","Saturday","Sunday"].index("Friday"),
                    'days_since_first': (datetime.now() - st.session_state.df['release_date'].min()).days,
                    'is_remix': 0,
                    'is_spedup': 0,
                    'artist_growth': 0.05,
                    'followers_per_market': base_followers / 185
                }
                
                # Optimized scenario
                opt_features = base_features.copy()
                opt_features.update({
                    'is_remix': 1 if test_version == "Remix" else 0,
                    'is_spedup': 1 if test_version == "Sped Up" else 0,
                    'release_dayofweek': ["Monday","Tuesday","Wednesday","Thursday",
                                         "Friday","Saturday","Sunday"].index(test_day)
                })
                
                # Calculate boosts
                promo_boost = {
                    "None": 1.0,
                    "Small": 1.15,
                    "Medium": 1.3,
                    "Large": 1.5,
                    "Massive": 2.0
                }[test_promo]
                
                collab_boost = {
                    "None": 1.0,
                    "A-List": 1.4,
                    "B-List": 1.25,
                    "C-List": 1.1,
                    "New Artist": 1.05
                }[test_collab]
                
                # Make predictions
                base_pred = np.expm1(st.session_state.model.predict(pd.DataFrame([base_features]))[0])
                opt_pred = np.expm1(st.session_state.model.predict(pd.DataFrame([opt_features]))[0])
                final_pred = int(opt_pred * promo_boost * collab_boost)
                
                # Calculate improvement
                improvement = final_pred - base_pred
                pct_improvement = (improvement / base_pred) * 100
                
                # Display results
                st.success("## Optimization Results")
                
                col1, col2 = st.columns(2)
                col1.metric("Baseline Prediction", f"{int(base_pred):,}")
                col2.metric("Optimized Prediction", f"{final_pred:,}",
                           delta=f"+{pct_improvement:.1f}%",
                           delta_color="inverse")
                
                # Show optimization components
                st.subheader("Improvement Breakdown")
                
                fig = go.Figure()
                
                components = {
                    "Version Type": opt_pred - base_pred,
                    "Promotion": (promo_boost - 1) * opt_pred,
                    "Collaboration": (collab_boost - 1) * opt_pred
                }
                
                fig.add_trace(go.Bar(
                    x=list(components.keys()),
                    y=list(components.values()),
                    marker_color=['#3498db', '#2ecc71', '#9b59b6']
                ))
                
                fig.update_layout(
                    title="Stream Gain Components",
                    yaxis_title="Additional Streams",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Gwamz Analytics Pro** v2.1 | [GitHub Repo](https://github.com/yourusername/gwamz-predictor) | [Report Issue](https://github.com/yourusername/gwamz-predictor/issues)
""")
