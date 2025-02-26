import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from BlackScholes import BlackScholes

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 15px;
    width: auto;
    margin: 0 auto;
}

.metric-call {
    background-color: #90ee90;
    color: #000000;
    margin-right: 10px;
    border-radius: 10px;
    padding: 15px;
    width: 100%;
}

.metric-put {
    background-color: #ffcccb;
    color: #000000;
    border-radius: 10px;
    padding: 15px;
    width: 100%;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin: 5px 0;
    color: #000000;
    text-align: center;
}

.metric-pnl {
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
}

.metric-pnl-positive {
    color: #006400;
}

.metric-pnl-negative {
    color: #8b0000;
}

.metric-label {
    font-size: 1.2rem;
    margin-bottom: 8px;
    color: #000000;
    text-align: center;
    font-weight: 600;
}

.risk-metrics-container {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.risk-card {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 20px;
    flex: 1;
}

.risk-title {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 20px;
    color: white;
}

.risk-metric {
    margin-bottom: 20px;
}

.risk-label {
    color: #a0a0a0;
    font-size: 1.1rem;
    margin-bottom: 5px;
}

.risk-value {
    color: #4CAF50;
    font-size: 1.5rem;
    font-weight: bold;
}

.risk-inf {
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs
with st.sidebar:
    st.title("Black-Scholes Model")
    st.write("`Made by: Christian Lee`")

    # Market parameters
    st.subheader("Market Parameters")
    current_price = st.number_input("Current Asset Price", min_value=0.01, value=100.0, step=0.01)
    strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=0.01)
    
    # Time parameters
    st.subheader("Time Parameters")
    time_to_maturity = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.01)
    purchase_time_to_maturity = st.number_input("Time to Maturity at Purchase (Years)", min_value=0.01, value=1.0, step=0.01)
    
    # Other parameters
    st.subheader("Other Parameters")
    volatility = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.20, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    
    # Purchase details
    st.subheader("Purchase Details")
    call_purchase_price = st.number_input("Call Purchase Price", min_value=0.0, value=0.0, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", min_value=0.0, value=0.0, step=0.01)

    # Heatmap parameters
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.number_input('Min Volatility', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.number_input('Max Volatility', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

# Main content
st.title("Black-Scholes Option Pricing Model")

# Initialize model and calculate values
bs_model = BlackScholes(
    time_to_maturity=time_to_maturity,
    strike=strike,
    current_price=current_price,
    volatility=volatility,
    interest_rate=interest_rate,
    call_purchase_price=call_purchase_price,
    put_purchase_price=put_purchase_price,
    purchase_time_to_maturity=purchase_time_to_maturity
)

call_price, put_price = bs_model.calculate_prices()
risk_metrics = bs_model.calculate_risk_metrics()

# Display metrics
col1, col2 = st.columns(2)

with col1:
    call_pnl = getattr(bs_model, 'call_pnl', 0.0)
    call_pnl_percentage = getattr(bs_model, 'call_pnl_percentage', 0.0)
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-call">
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
                <div class="metric-pnl {'metric-pnl-positive' if call_pnl > 0 else 'metric-pnl-negative'}">
                    PnL: ${call_pnl:.2f} ({call_pnl_percentage:.1f}%)
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    put_pnl = getattr(bs_model, 'put_pnl', 0.0)
    put_pnl_percentage = getattr(bs_model, 'put_pnl_percentage', 0.0)
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-put">
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
                <div class="metric-pnl {'metric-pnl-positive' if put_pnl > 0 else 'metric-pnl-negative'}">
                    PnL: ${put_pnl:.2f} ({put_pnl_percentage:.1f}%)
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Risk metrics
st.subheader("Risk Metrics")

metrics_html = f"""
<div class="risk-metrics-container">
    <div class="risk-card">
        <div class="risk-title">Call Option</div>
        <div class="risk-metric">
            <div class="risk-label">Maximum Loss:</div>
            <div class="risk-value">${risk_metrics['max_call_loss']:.2f}</div>
        </div>
        <div class="risk-metric">
            <div class="risk-label">Maximum Gain:</div>
            <div class="risk-value risk-inf">{"$inf" if risk_metrics['max_call_gain'] == float('inf') else f"${risk_metrics['max_call_gain']:.2f}"}</div>
        </div>
        <div class="risk-metric">
            <div class="risk-label">Break-even:</div>
            <div class="risk-value">${risk_metrics['call_breakeven']:.2f}</div>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-title">Put Option</div>
        <div class="risk-metric">
            <div class="risk-label">Maximum Loss:</div>
            <div class="risk-value">${risk_metrics['max_put_loss']:.2f}</div>
        </div>
        <div class="risk-metric">
            <div class="risk-label">Maximum Gain:</div>
            <div class="risk-value">${risk_metrics['max_put_gain']:.2f}</div>
        </div>
        <div class="risk-metric">
            <div class="risk-label">Break-even:</div>
            <div class="risk-value">${risk_metrics['put_breakeven']:.2f}</div>
        </div>
    </div>
</div>
"""

st.markdown(metrics_html, unsafe_allow_html=True)

def plot_heatmap(bs_model, spot_range, vol_range, strike, purchase_price=None, type='value'):
    values = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate,
                call_purchase_price=bs_model.call_purchase_price,
                put_purchase_price=bs_model.put_purchase_price,
                purchase_time_to_maturity=bs_model.purchase_time_to_maturity
            )
            call_price, put_price = bs_temp.calculate_prices()
            
            if type == 'value_call':
                values[i, j] = call_price
            elif type == 'value_put':
                values[i, j] = put_price
            elif type == 'pnl_call' and purchase_price is not None:
                values[i, j] = call_price - purchase_price
            elif type == 'pnl_put' and purchase_price is not None:
                values[i, j] = put_price - purchase_price
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'value' in type:
        cmap = 'YlOrRd'
    else:
        cmap = sns.diverging_palette(10, 120, as_cmap=True)
    
    sns.heatmap(
        values,
        xticklabels=np.round(spot_range, 2),
        yticklabels=np.round(vol_range, 2),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0 if 'pnl' in type else None,
        ax=ax
    )
    
    title = f"{'CALL' if 'call' in type else 'PUT'} {'Value' if 'value' in type else 'PnL'}"
    ax.set_title(title)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    return fig

# Option Values and PnL Heatmaps
st.title("Option Values and PnL Heatmaps")
st.info("Explore how option values and PnL change with varying spot prices and volatility levels.")

spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)

# Call Option Heatmaps
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Value")
    fig_call_value = plot_heatmap(bs_model, spot_range, vol_range, strike, type='value_call')
    st.pyplot(fig_call_value)

with col2:
    st.subheader("Call Option PnL")
    fig_call_pnl = plot_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, type='pnl_call')
    st.pyplot(fig_call_pnl)

# Put Option Heatmaps
col1, col2 = st.columns(2)
with col1:
    st.subheader("Put Option Value")
    fig_put_value = plot_heatmap(bs_model, spot_range, vol_range, strike, type='value_put')
    st.pyplot(fig_put_value)

with col2:
    st.subheader("Put Option PnL")
    fig_put_pnl = plot_heatmap(bs_model, spot_range, vol_range, strike, put_purchase_price, type='pnl_put')
    st.pyplot(fig_put_pnl)

# Footer notes
st.markdown("""
---
### Notes:
- The heatmaps show how option values and PnL change with different spot prices and volatility levels
- For PnL heatmaps: 
  - Green indicates positive PnL (profit)
  - Red indicates negative PnL (loss)
  - White/neutral indicates break-even points
- All calculations assume constant interest rates and time to maturity
""")