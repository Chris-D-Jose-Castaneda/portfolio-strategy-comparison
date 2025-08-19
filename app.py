"""
SPX Options Risk Dashboard App

To run this Streamlit app:
1. Install Streamlit:
   pip install streamlit
2. Launch the app from your project directory:
   streamlit run app.py
   -- or if that command isn't recognized: --
   python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import yfinance as yf

# Constants
DATA_PATH = 'SPX_options_reference_chain_20150101_20250717.csv'
CONTRACT_MULT = 100

# Black‑Scholes Greeks function
def bs_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type.lower() == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    return delta, gamma, vega

# Data loading and Greek exposure computation
def compute_exposures(spot, snapshot_date, df_chain, sigma, risk_free):
    # Slice latest snapshot
    df = df_chain[df_chain['snapshot_date'] == snapshot_date].copy()
    df['days_to_expiry'] = (df['expiration_date'] - snapshot_date).dt.days
    df['T'] = df['days_to_expiry'] / 365.0
    
    exposures = []
    for _, row in df.iterrows():
        d, g, v = bs_greeks(
            spot,
            row['strike_price'],
            max(row['T'], 1e-6),
            risk_free,
            sigma,
            option_type=row['option_type']
        )
        oi = row.get('open_interest', 1)
        exposures.append({
            'expiration': row['expiration_date'],
            'delta': d * oi * CONTRACT_MULT,
            'gamma': g * oi * CONTRACT_MULT,
            'vega' : v * oi * CONTRACT_MULT
        })
    df_risk = pd.DataFrame(exposures)
    agg = df_risk.groupby('expiration').sum().reset_index()
    return agg

# P&L simulation function
def simulate_pnl(agg, spot, shock_pct, vol_shift, sims):
    # Portfolio exposures
    port_d = agg['delta'].sum()
    port_g = agg['gamma'].sum()
    port_v = agg['vega'].sum()
    dS = spot * shock_pct
    dv = vol_shift
    # Analytical P&L samples
    base_pnl = port_d * dS + 0.5 * port_g * dS**2 + port_v * dv
    pnl_hedged = np.full(sims, base_pnl)
    # Placeholder for costs (no costs)
    pnl_tc = pnl_hedged.copy()
    # VaR
    var95 = -np.percentile(pnl_hedged, 5)
    var99 = -np.percentile(pnl_hedged, 1)
    return pnl_hedged, pnl_tc, var95, var99

# Main Streamlit app
def main():
    st.title('SPX Options Risk Dashboard')

    # Sidebar for dynamic parameters
    st.sidebar.header('Model Parameters')
    # Instrument selection
    instrument = st.sidebar.selectbox(
        'Underlying Instrument',
        options=['^GSPC', 'SPY', '^VIX'],
        index=0
    )
    sigma      = st.sidebar.slider('Implied Volatility', 0.05, 1.0, 0.2, 0.01)
    risk_free  = st.sidebar.slider('Risk‑free Rate', 0.0, 0.05, 0.01, 0.001)
    shock_pct  = st.sidebar.slider('Spot Shock (%)', 0.001, 0.05, 0.01, 0.001)
    vol_shift  = st.sidebar.slider('Vol Shift', 0.0, 0.5, 0.1, 0.01)
    sims       = st.sidebar.number_input('Simulations', min_value=100, max_value=20000, value=5000, step=100)

    # Load chain once
    df_chain = pd.read_csv(DATA_PATH, parse_dates=['snapshot_date','expiration_date'])
    snapshot_date = df_chain['snapshot_date'].max()

    # Fetch spot price with fallback
    try:
        spot = yf.Ticker(instrument).history(period='1d')['Close'].iloc[-1]
    except Exception:
        spot = df_chain['strike_price'].median()
        st.sidebar.warning(f'Using fallback spot={spot:.2f}')

    agg = compute_exposures(spot, snapshot_date, df_chain, sigma, risk_free)

    # Display exposures
    st.subheader('Aggregated Greek Exposures')
    fig = go.Figure()
    for greek in ['delta','gamma','vega']:
        fig.add_trace(go.Bar(
            x=agg['expiration'].dt.strftime('%Y-%m-%d'),
            y=agg[greek],
            name=greek.capitalize()
        ))
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Simulate P&L
    pnl_hedged, pnl_tc, var95, var99 = simulate_pnl(agg, spot, shock_pct, vol_shift, sims)

    # P&L distributions
    st.subheader('P&L Distributions')
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Delta‑Hedged','With Costs'])
    fig2.add_trace(go.Histogram(x=pnl_hedged, name='Delta‑Hedged'), row=1, col=1)
    fig2.add_trace(go.Histogram(x=pnl_tc,     name='With Costs'),    row=1, col=2)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # VaR metrics
    st.subheader('Value at Risk')
    st.markdown(f'**95% VaR:** {var95:,.0f}')
    st.markdown(f'**99% VaR:** {var99:,.0f}')

if __name__ == '__main__':
    main()

# Run with: python -m streamlit run app.py
