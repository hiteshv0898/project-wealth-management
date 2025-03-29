import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gc
from utils import get_live_data
from analysis import PortfolioAnalyzer
from sentiment import analyze_sentiment, fetch_financial_news

# Memory management

def clear_memory():
    gc.collect()
    print("✅ Memory cleared using garbage collection.")

# Dashboard Title
st.set_page_config(page_title='Wealth Management Dashboard', layout='wide')
st.title('📊 Wealth Management Dashboard')

# Portfolio Section
st.sidebar.header('Portfolio Overview')
allocation = {'AAPL': 30, 'TSLA': 30, 'GOOGL': 20, 'MSFT': 20}
initial_capital = 100000
portfolio_data = pd.DataFrame([{'Asset': k, 'Investment': v * initial_capital / 100} for k, v in allocation.items()])
st.sidebar.write(portfolio_data)

# Live Data
st.header('📈 Live Market Prices')
def fetch_live_data():
    data = []
    for asset in allocation.keys():
        live_price = get_live_data(asset).history(period='1d')['Close'].iloc[-1]
        data.append({'Asset': asset, 'Live Price': live_price})
    clear_memory()
    return pd.DataFrame(data)

if st.button('Fetch Live Data'):
    live_data = fetch_live_data()
    st.write(live_data)

# Portfolio Analysis
st.header('📊 Portfolio Analysis')
analyzer = PortfolioAnalyzer(portfolio_data, risk_tolerance='moderate')
st.write('Sharpe Ratio and Risk-Adjusted Returns:')
analyzer.perform_analysis()

# Sentiment Analysis
st.header('📰 Sentiment Analysis')
def show_sentiment(asset):
    news = fetch_financial_news(asset)
    sentiment_score = analyze_sentiment(news)
    st.write(f"**{asset} Sentiment Score:** {sentiment_score}")
    for article in news:
        st.write(f"- {article}")
    clear_memory()

asset_choice = st.selectbox('Select an Asset for Sentiment Analysis', list(allocation.keys()))
if st.button('Analyze Sentiment'):
    show_sentiment(asset_choice)

# Portfolio Visualization
st.header('📊 Portfolio Allocation')
fig = px.pie(values=portfolio_data['Investment'], names=portfolio_data['Asset'], title='Portfolio Allocation')
st.plotly_chart(fig)

# Footer
st.markdown('---')
st.write('💡 Developed as part of the Wealth Management Agent Project.')
