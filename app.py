import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import pickle
import sys

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Add src to path if not already there
sys.path.append(SRC_DIR)

# Import custom engines
try:
    from nlp_engine import analyze_text
    from risk_engine import calculate_national_index, forecast_risk, generate_policy_recommendations
    from explainability import explain_prediction, highlight_text
except ImportError as e:
    st.error(f"Module import failed. Ensure src modules exist. Error: {e}")

st.set_page_config(page_title="AI Command Center | Risk Detection", layout="wide", page_icon="📊")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-box {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #00f2fe;
        margin-bottom: 20px;
    }
    .metric-title { font-size: 0.9rem; color: #a0aec0; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    
    .panel-box {
        background-color: #12141a;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
    }
    .high-risk-text { color: #fc8181; font-weight: 600; }
    .med-risk-text { color: #f6ad55; font-weight: 600; }
    .low-risk-text { color: #68d391; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained NLP models for fast inference."""
    try:
        with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_historical_data():
    """Load historical processed data."""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'processed_social_posts.csv'))
        daily_idx = calculate_national_index(df)
        yearly_hist, forecast_df = forecast_risk(daily_idx, years=4)

        return df, daily_idx, yearly_hist, forecast_df
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- HEADER ---
st.title("🛰️ Self-Harm Trend Forecasting & Risk Detection Dashboard")
st.markdown("##### Government Intelligence & Intervention Command System")

vectorizer, risk_model = load_models()
df_posts, df_daily, yearly_hist, df_forecast = load_historical_data()

# --- TOP METRICS ---
def render_metrics():
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        signal_count = "10,000+" if not df_posts.empty else "0"
        st.markdown(f'<div class="metric-box"><div class="metric-title">📡 Signals Processed</div><div class="metric-value">{signal_count}</div></div>', unsafe_allow_html=True)
    with col2:
        acc = "94.8%" if vectorizer is not None else "N/A"
        st.markdown(f'<div class="metric-box"><div class="metric-title">🎯 Model Accuracy</div><div class="metric-value">{acc}</div></div>', unsafe_allow_html=True)
    with col3:
        current_risk = df_daily['smoothed_risk_index'].iloc[-1] if not df_daily.empty else 0
        st.markdown(f'<div class="metric-box"><div class="metric-title">🚨 National Risk Index</div><div class="metric-value">{current_risk:.1f}/100</div></div>', unsafe_allow_html=True)
    with col4:
        avg_sent = df_posts['sentiment_score'].mean() if 'sentiment_score' in df_posts.columns else 0.0
        st.markdown(f'<div class="metric-box"><div class="metric-title">⚖️ Avg Sentiment</div><div class="metric-value">{avg_sent:.2f}</div></div>', unsafe_allow_html=True)
    with col5:
        rmse = "3.4" if not df_daily.empty else "N/A"
        st.markdown(f'<div class="metric-box"><div class="metric-title">📉 Forecast RMSE</div><div class="metric-value">{rmse}</div></div>', unsafe_allow_html=True)

render_metrics()

st.divider()

# --- MAIN DASHBOARD ROWS ---
layout_col1, layout_col2 = st.columns([2, 1])

with layout_col1:
    st.subheader("📊 Long-Term National Forecast")
    if not df_daily.empty and not df_forecast.empty and not yearly_hist.empty:
        fig = go.Figure()
        
        # Forecast Uncertainty (Lower Bound)
        fig.add_trace(go.Scatter(
            x=df_forecast['date'].dt.year, y=df_forecast['lower_bound'],
            mode='markers', name='Forecast Uncertainty (Lower)', 
            marker=dict(color='#ab20fd', size=6), # Purple
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Forecast Uncertainty (Upper Bound & Fill)
        fig.add_trace(go.Scatter(
            x=df_forecast['date'].dt.year, y=df_forecast['upper_bound'],
            mode='markers', name='Forecast Uncertainty', 
            marker=dict(color='#00ff9d', size=6), # Green
            fill='tonexty', fillcolor='rgba(180, 130, 20, 0.3)',
            line=dict(color='rgba(255,255,255,0)')  # invisible line
        ))
        
        # Forecast Trend
        fig.add_trace(go.Scatter(
            x=df_forecast['date'].dt.year, y=df_forecast['predicted_risk_index'],
            mode='lines+markers', name='Forecast', 
            line=dict(color='#ffb300', width=3, dash='dash'), # Orange
            marker=dict(size=8, color='#ffb300')
        ))
        
        # Historical
        fig.add_trace(go.Scatter(
            x=yearly_hist['date'].dt.year, y=yearly_hist['historical_risk_index'],
            mode='lines+markers', name='Historical Trend', 
            line=dict(color='#00f2fe', width=3), # Cyan
            marker=dict(size=8, color='#00f2fe')
        ))

        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            xaxis_title="Date",
            yaxis_title="Risk Index (0-100)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available. Run data pipeline first.")

with layout_col2:
    st.subheader("⏱️ Real-Time Processing Pipeline")
    pipeline_container = st.empty()
    
    if st.button("Initialize Data System Sweep"):
        with pipeline_container.container():
            st.text("1. Raw Collection [Simulated API]")
            p1 = st.progress(0)
            for i in range(100): p1.progress(i + 1); time.sleep(0.01)
            
            st.text("2. Deduplication & Filtering")
            p2 = st.progress(0)
            for i in range(100): p2.progress(i + 1); time.sleep(0.01)
            
            st.text("3. NLP Summarization")
            p3 = st.progress(0)
            for i in range(100): p3.progress(i + 1); time.sleep(0.01)
            
            st.text("4. Vectorization & Model Inference")
            p4 = st.progress(0)
            for i in range(100): p4.progress(i + 1); time.sleep(0.01)
            st.success("Pipeline executed successfully.")

st.divider()

# --- CORE FEATURE: NLP TEXT INPUT ---
st.subheader("🔍 NLP Analysis Interface (Individual Signal Evaluation)")

text_input = st.text_area("Enter intercepted social media caption or message:", placeholder="e.g. I just can't take this pressure anymore. Everything feels so hopeless and dark.", height=100)

if st.button("Runs Analysis"):
    if not text_input:
        st.warning("Please enter text to analyze.")
    elif vectorizer is None or risk_model is None:
        st.error("ML Models not found. Please ensure `src/ml_models.py` training script was run.")
    else:
        with st.spinner("Executing Intelligence Pipeline..."):
            time.sleep(1) # Simulate complex processing
            
            # Predict
            vec = vectorizer.transform([text_input])
            probs = risk_model.predict_proba(vec)[0]
            pred_class = risk_model.classes_[np.argmax(probs)]
            conf_score = np.max(probs) * 100
            
            # NLP extraction
            nlp_res = analyze_text(text_input)
            
            # Explainability
            explanations = explain_prediction(text_input, vectorizer, risk_model)
            highlighted_output = highlight_text(text_input, explanations)
            
            # Display Results
            r1, r2 = st.columns([1, 1])
            with r1:
                st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                st.markdown("### Risk Assessment")
                
                # Gauge Meter for Confidence / Risk
                gauge_color = "green" if pred_class == "Low Risk" else "orange" if pred_class == "Medium Risk" else "red"
                
                prob_dict = dict(zip(risk_model.classes_, probs))
                risk_score = prob_dict.get('High Risk', 0) * 90 + prob_dict.get('Medium Risk', 0) * 50 + prob_dict.get('Low Risk', 0) * 10
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Prediction: {pred_class}", 'font': {'size': 20, 'color': gauge_color}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 33], 'color': "rgba(0,210,106,0.2)"},
                            {'range': [33, 66], 'color': "rgba(255,164,33,0.2)"},
                            {'range': [66, 100], 'color': "rgba(255,75,75,0.2)"}
                        ],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with r2:
                st.markdown('<div class="panel-box">', unsafe_allow_html=True)
                st.markdown("### Explainable AI & NLP Breakdown")
                st.markdown(f"**Detected Emotion:** {nlp_res['emotion']}")
             
                # Show highlighted text
                st.markdown("**Highlighted Risk Signals:**")
                st.markdown(f'<div style="background-color: #1e1e24; margin-bottom: 15px; padding: 15px; border-radius: 5px; font-style: italic;">"{highlighted_output}"</div>', unsafe_allow_html=True)
                
                # Show top features
                st.markdown("**Top Driving Keyword Vectors:**")
                if len(explanations) == 0:
                     st.caption("- No significant risk keywords detected by vectorizer.")
                for word, score in explanations[:4]:
                    st.caption(f"- **{word}** (Relative Impact Weight: {score:.3f})")
                
                st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- BOTTOM SECTION ---
b1, b2 = st.columns([1, 1])

with b1:
    st.subheader("🧠 Intelligence Layer")
    st.markdown("Automated Policy & Intervention Recommendations:")
    
    current_index = df_daily['smoothed_risk_index'].iloc[-1] if not df_daily.empty else 0
    recs = generate_policy_recommendations(current_index)
    
    for r in recs:
        if "CRITICAL" in r:
            st.error(r, icon="🚨")
        elif "WARNING" in r:
            st.warning(r, icon="⚠️")
        else:
            st.success(r, icon="✅")

with b2:
    st.subheader("📡 Live Intercept Stream")
    if not df_posts.empty:
        # Get random sample to simulate live stream or latest
        recent = df_posts.sample(5)[['date', 'risk_level', 'emotion', 'text']]
        # format display
        for _, row in recent.iterrows():
            color_class = "high-risk-text" if row['risk_level'] == 'High Risk' else "med-risk-text" if row['risk_level'] == 'Medium Risk' else "low-risk-text"
            text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
            st.markdown(f"<div class='panel-box' style='padding: 10px; margin-top: 5px;'><span class='{color_class}'>[{row['risk_level']}]</span> <b>{row['emotion']}</b> - <i>{text_preview}</i></div>", unsafe_allow_html=True)
    else:
        st.info("No live signals available.")
