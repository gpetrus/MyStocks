# pip install streamlit
# pip install yfinance scikit-learn
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import Stocks_Modules
from Stocks_Modules import *

# Streamlit Dashboard
st.set_page_config(layout="wide")
st.title("My Stocks Dashboard")

ticker_input = st.text_input("Enter stock ticker symbols (comma-separated, e.g., AAPL, MSFT, GOOGL)", "AAPL, UNH, COST")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Select years of history
years_input = st.slider("Select number of years of history", min_value=1, max_value=30, value=10)

# Select which charts to show
st.markdown("### Select Charts to Display:")
# Toggle to select all charts
select_all = st.checkbox("Select All Charts", value=True)
# First row of checkboxes
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    show_dividends = st.checkbox("Dividend History", value=select_all, key="dividends_cb")
with col2:
    show_pe = st.checkbox("P/E Ratio", value=select_all, key="pe_cb")
with col3:
    show_revenue = st.checkbox("Revenue", value=select_all, key="revenue_cb")
with col4:
    show_net_income = st.checkbox("Net Income", value=select_all, key="net_cb")
with col5:
    show_price_ma = st.checkbox("Price + MAs", value=select_all, key="price_ma_cb")
with col6:
    show_vs_sp500 = st.checkbox("Price vs. S&P 500", value=select_all, key="sp500_cb")    
with col7:
    show_valuation_ml = st.checkbox("Valuation Anomaly (ML)", value=select_all, key="val_anomaly")    

if st.button("Show Charts"):
    tabs = st.tabs(tickers)

    for tab, ticker in zip(tabs, tickers):
        with tab:
            st.markdown(f"## {ticker}")
            show_key_metrics(ticker) 

            columns = []
            if show_dividends:
                columns.append("dividend")
            if show_pe:
                columns.append("pe")
            if show_revenue:
                columns.append("revenue")
            if show_net_income:
                columns.append("net_income")
            if show_price_ma:
                columns.append("price_ma")
            if show_price_ma:
                columns.append("price_vs_sp500")  
            if show_valuation_ml:
                columns.append("valuation_ml")              
            
            # Make columns layout dynamic based on selection
            chart_cols = st.columns(len(columns)) if columns else []

            for i, chart_type in enumerate(columns):
                with chart_cols[i]:
                    if chart_type == "dividend":
                        st.subheader("Dividend History")
                        plot_dividend_history(ticker, years_back=years_input)
                    elif chart_type == "pe":
                        st.subheader("P/E Ratio")
                        plot_pe_ratio(ticker, years_back=years_input)
                    elif chart_type == "revenue":
                        st.subheader("Revenue")
                        plot_revenue(ticker)
                    elif chart_type == "net_income":
                        st.subheader("Net Income")
                        plot_net_income(ticker)
                    elif chart_type == "price_ma":
                        st.subheader("Stock Price with MAGs")
                        plot_price_with_moving_averages(ticker, period="60mo")
                    elif chart_type == "price_vs_sp500":
                        st.subheader("Price vs. S&P 500")
                        plot_price_vs_sp500(ticker, period="1y")
                    elif chart_type == "valuation_ml":
                        detect_valuation_anomalies(ticker)
