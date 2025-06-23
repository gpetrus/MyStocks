import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime, timedelta, timezone

def display_ticker_news(ticker: str, max_items: int = 5):
    st.subheader("📰 Latest News")

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        if not news_items:
            st.info("No news available for this ticker.")
            return

        # Limit to the latest N articles
        for item in news_items[:max_items]:
            title = item.get("title", "Untitled")
            link = item.get("link", "#")
            publisher = item.get("publisher", "Unknown")
            st.markdown(
                f"- **[{title}]({link})**  <sub><i>{publisher}</i></sub>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.warning(f"Failed to retrieve news for {ticker}.")
        st.caption(f"Debug info: {e}")

def display_upcoming_events(ticker):

    try:
        st.subheader("📅 Upcoming Events")
        stock = yf.Ticker(ticker)

        # Earnings Date from earnings_dates
        df = stock.earnings_dates
        df.index = df.index.tz_convert(None)
        future = df[df.index >= pd.Timestamp.today()] if not df.empty else pd.DataFrame()

        if not future.empty:
            next_earn = future.index[0]
            eps_est = future.iloc[0]["EPS Estimate"]
            st.markdown(f"**Earnings Date:** 📢 {next_earn.strftime('%B %d, %Y')} (EPS est: {eps_est})")
        else:
            st.markdown("**Earnings Date:** Not available")

        # Ex-dividend
        actions = stock.actions
        future_divs = actions[
            (actions.index.tz_localize(None) > pd.Timestamp.today()) &
            (actions['Dividends'] > 0)
        ]
        if not future_divs.empty:
            next_ex_div = future_divs.index[0]
            st.markdown(f"**Ex-Dividend Date:** 💰 {next_ex_div.strftime('%B %d, %Y')}")
        else:
            st.markdown("**Ex-Dividend Date:** Not available")

    except Exception as e:
        st.warning(f"Could not retrieve event data for {ticker}.")
        st.caption(f"Debug info: {e}")


def plot_dividend_history(ticker_symbol, years_back=10):
    stock = yf.Ticker(ticker_symbol)
    dividends = stock.dividends

    if dividends.empty:
        st.warning(f"No dividend data available for {ticker_symbol}.")
        return

    # Filter dividends by date range
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=years_back)
    dividends.index = dividends.index.tz_localize(None)
    cutoff_date = cutoff_date.tz_localize(None)
    dividends = dividends.loc[dividends.index >= cutoff_date]

    if dividends.empty:
        st.warning(f"No dividend data for {ticker_symbol} in the last {years_back} years.")
        return

    # Resample quarterly totals
    quarterly_dividends = dividends.resample('QE').sum()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(quarterly_dividends.index, quarterly_dividends.values, width=50, color='skyblue')
    ax.set_title(f"Quarterly Dividend History for {ticker_symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Dividends (USD)")
    plt.xticks(rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig,use_container_width=True)

def plot_pe_ratio(ticker_symbol, years_back=10):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=f"{years_back}y")

    if hist.empty or 'Close' not in hist:
        st.warning(f"No price data available for {ticker_symbol}.")
        return

    eps_ttm = stock.info.get("trailingEps", None)

    if eps_ttm is None or eps_ttm == 0:
        st.warning(f"No EPS data available for {ticker_symbol}.")
        return

    # Calculate PE ratio
    pe_ratio = hist['Close'] / eps_ttm

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pe_ratio.index, pe_ratio.values, label='P/E Ratio', color='orange')
    ax.set_title(f"Price-to-Earnings (P/E) Ratio for {ticker_symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("P/E Ratio")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig,use_container_width=True)

def plot_revenue(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try:
        revenue = stock.quarterly_financials.loc["Total Revenue"]
    except KeyError:
        st.warning(f"No revenue data available for {ticker_symbol}.")
        return

    if revenue.empty:
        st.warning(f"No revenue data available for {ticker_symbol}.")
        return

    # Sort by date ascending
    revenue = revenue.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(revenue.index, revenue.values / 1e9, color='mediumslateblue')  # Convert to billions
    ax.set_title(f"Annual Revenue History for {ticker_symbol}")
    ax.set_xlabel("Fiscal Year End")
    ax.set_ylabel("Revenue (Billion USD)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig,use_container_width=True)

def plot_net_income(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)

    try:
        net_income = stock.financials.loc["Net Income"]
    except KeyError:
        st.warning(f"No net income data available for {ticker_symbol}.")
        return

    if net_income.empty:
        st.warning(f"No historical net income data found.")
        return

    # Sort by date ascending
    net_income = net_income.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(net_income.index, net_income.values / 1e9, color='darkred')  # Convert to billions
    ax.set_title(f"Annual Net Income for {ticker_symbol}")
    ax.set_xlabel("Fiscal Year End")
    ax.set_ylabel("Net Income (Billion USD)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig,use_container_width=True)

def plot_price_with_moving_averages(ticker_symbol, period, interval="1d"):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=period, interval=interval)

    if hist.empty:
        st.warning(f"No historical price data available for {ticker_symbol}.")
        return

    # Calculate moving averages
    hist["MA_5"] = hist["Close"].rolling(window=5).mean()
    hist["MA_20"] = hist["Close"].rolling(window=20).mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist.index, hist["Close"], label="Close Price", color="black")
    ax.plot(hist.index, hist["MA_5"], label="5-Day MA", color="blue", linestyle="--")
    ax.plot(hist.index, hist["MA_20"], label="20-Day MA", color="orange", linestyle="--")

    ax.set_title(f"{ticker_symbol} Stock Price with 5/20-Day Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig,use_container_width=True)

def plot_price_vs_sp500(ticker_symbol, period="1y"):
    stock = yf.Ticker(ticker_symbol)
    sp500 = yf.Ticker("^GSPC")

    stock_history = stock.history(period=period)
    sp500_history = sp500.history(period=period)

    if stock_history.empty or sp500_history.empty:
        st.warning("Unable to retrieve stock or S&P 500 data.")
        return

    # Normalize prices to start at 100
    stock_price = stock_history['Close'] / stock_history['Close'].iloc[0] * 100
    sp500_price = sp500_history['Close'] / sp500_history['Close'].iloc[0] * 100

    # Align indexes just in case
    df = pd.DataFrame({
        ticker_symbol: stock_price,
        "S&P 500": sp500_price
    }).dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[ticker_symbol], label=ticker_symbol)
    ax.plot(df.index, df["S&P 500"], label="S&P 500", linestyle='--', color='gray')
    ax.set_title(f"{ticker_symbol} vs. S&P 500 ({period} Performance)")
    ax.set_ylabel("Normalized Price (100 = Start)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig,use_container_width=True)


def show_key_metrics(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    try:
        price = info.get("currentPrice", "N/A")
        market_cap = info.get("marketCap", None)
        if market_cap:
            market_cap_display = f"${market_cap / 1e9:.2f}B"
        else:
            market_cap_display = "N/A"
    except Exception:
        price = "N/A"
        market_cap_display = "N/A"

    # Show metrics in columns
    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${price}")
    col2.metric("Market Cap", market_cap_display)

def detect_valuation_anomalies(ticker_symbol, years_back=5):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=f"{years_back}y")
    fin = stock.financials
    info = stock.info

    if hist.empty: # or not fin.any():
        st.warning("Insufficient data for anomaly detection.")
        return

    # Build feature set
    pe_ratio = info.get("trailingPE", np.nan)
    ps_ratio = info.get("priceToSalesTrailing12Months", np.nan)
    market_cap = info.get("marketCap", np.nan)
    revenue = info.get("totalRevenue", np.nan)
    net_income = info.get("netIncomeToCommon", np.nan)

    features = np.array([[pe_ratio, ps_ratio, market_cap, revenue, net_income]])

    # Use historical range to set realistic bounds (or train model on multiple stocks)
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    # Fake historical data (ideally from a sector peer group or market)
    dummy_data = features + np.random.normal(0, 0.1, (100, 5))
    iso_model.fit(dummy_data)

    anomaly_score = iso_model.decision_function(features)[0]
    is_anomaly = iso_model.predict(features)[0] == -1

    # Generate explanation
    explanations = []

    if pe_ratio and pe_ratio > np.nanmean(dummy_data[:, 0]) * 1.5:
        explanations.append(f"Unusually high P/E ratio ({pe_ratio:.2f})")

    if ps_ratio and ps_ratio > np.nanmean(dummy_data[:, 1]) * 1.5:
        explanations.append(f"Elevated price-to-sales ratio ({ps_ratio:.2f})")

    if revenue and net_income and net_income < 0:
        explanations.append("Company is currently unprofitable despite positive revenue.")

    if not explanations:
        explanations.append("Valuation appears within normal range based on available metrics.")

    status = "⚠️ Potential Over/Undervaluation Detected" if is_anomaly else "✅ Valuation looks typical"

    # Output
    st.subheader("Valuation Anomaly Analysis")
    st.markdown(f"**{status}**")
    for ex in explanations:
        st.markdown(f"- {ex}")
