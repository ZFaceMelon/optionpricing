import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Black-Scholes function
def black_scholes(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call and put prices
    call_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    put_price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# Streamlit app
st.title("Black-Scholes Option Pricing Model & Heatmaps")

# Add custom CSS for margins
st.markdown("""
    <style>
        .price-box {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .heatmap-container {
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    S = st.number_input("Stock Price (S)", min_value=0.0, value=100.0)
    X = st.number_input("Strike Price (X)", min_value=0.0, value=100.0)
    T = st.number_input("Time to Maturity (T in years)", min_value=0.0, value=1.0)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
    sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.2)

    st.subheader("Heatmap Range")
    min_spot_price = st.number_input("Min Spot Price", min_value=0.0, value=50.0)
    max_spot_price = st.number_input("Max Spot Price", min_value=0.0, value=150.0)
    min_volatility = st.number_input("Min Volatility (σ)", min_value=0.0, value=0.1)
    max_volatility = st.number_input("Max Volatility (σ)", min_value=0.0, value=0.5)

# Fixed number of points for heatmap
num_points = 10

# Create grid of spot prices and volatilities
spot_prices = np.linspace(min_spot_price, max_spot_price, num_points)
volatilities = np.linspace(min_volatility, max_volatility, num_points)

# Initialize matrices for call and put prices
call_prices = np.zeros((num_points, num_points))
put_prices = np.zeros((num_points, num_points))

# Calculate option prices for each combination of spot price and volatility
for i, S_grid in enumerate(spot_prices):
    for j, sigma_grid in enumerate(volatilities):
        call_price, put_price = black_scholes(S_grid, X, T, r, sigma_grid)
        call_prices[j, i] = call_price  # Heatmap rows are volatility, cols are spot price
        put_prices[j, i] = put_price

# Calculate individual call and put prices for the user inputs
call_price, put_price = black_scholes(S, X, T, r, sigma)

# Display call and put prices in boxes with margin between them and the title
st.markdown('<div class="price-box">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Display call price in green box
    st.markdown(
        f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px; text-align: center;">'
        f'<strong>Call Option Price: ${call_price:.2f}</strong>'
        f'</div>', unsafe_allow_html=True
    )

with col2:
    # Display put price in red box
    st.markdown(
        f'<div style="background-color: #FF7F7F; padding: 10px; border-radius: 5px; text-align: center;">'
        f'<strong>Put Option Price: ${put_price:.2f}</strong>'
        f'</div>', unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Add margin between the boxes and heatmaps
st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)

# Plot the heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Heatmap for Call Prices
sns.heatmap(call_prices, ax=ax1, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap="YlGnBu")
ax1.set_title("Call Option Prices")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")

# Heatmap for Put Prices
sns.heatmap(put_prices, ax=ax2, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap="YlOrRd")
ax2.set_title("Put Option Prices")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")

# Display the heatmaps in Streamlit
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)
