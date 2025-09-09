# realtime_stock_dashboard.py
# Single-file Streamlit app — corrected & improved version

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

# --- Page config & CSS ---
st.set_page_config(
    page_title="Real-Time Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .positive { color: #00aa00; }
    .negative { color: #d40000; }
    .neutral { color: #ffa500; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Helpers / Classes ---


class StockDataFetcher:
    """Handles fetching stock data (lightweight, resilient)."""

    @staticmethod
    @st.cache_data(ttl=300)
    def get_realtime_data(symbols: List[str]) -> pd.DataFrame:
        """Fetch basic realtime-like metrics per symbol.

        Note: yfinance can be slow for many symbols. This loops ticker-by-ticker,
        but caches results for `ttl` seconds so UI stays responsive for a bit.
        """
        out = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            try:
                ticker = yf.Ticker(symbol)
                # Try intraday minute data (if available)
                try:
                    hist = ticker.history(period="1d", interval="1m")
                except Exception:
                    hist = pd.DataFrame()

                info = {}
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}

                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
                    open_price = float(hist["Open"].iloc[0]) if "Open" in hist.columns else current_price
                    high_price = float(hist["High"].max()) if "High" in hist.columns else current_price
                    low_price = float(hist["Low"].min()) if "Low" in hist.columns else current_price
                    volume = int(hist["Volume"].sum()) if "Volume" in hist.columns else int(info.get("volume", 0) or 0)
                else:
                    current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
                    open_price = float(info.get("open") or current_price or 0)
                    high_price = float(info.get("dayHigh") or current_price or 0)
                    low_price = float(info.get("dayLow") or current_price or 0)
                    volume = int(info.get("volume") or 0)

                change = current_price - open_price if open_price is not None else 0.0
                change_percent = (change / open_price * 100) if open_price not in (0, None) else 0.0

                stock_data = {
                    "Symbol": symbol,
                    "Company": info.get("longName", symbol),
                    "Price": current_price,
                    "Change": change,
                    "Change%": change_percent,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Volume": volume,
                    "Market Cap": int(info.get("marketCap") or 0),
                    "PE Ratio": info.get("trailingPE"),
                    "Sector": info.get("sector", "N/A"),
                }
                out.append(stock_data)
            except Exception as e:
                # keep the app alive if a ticker fails
                st.warning(f"Failed to fetch {symbol}: {e}")
                out.append(
                    {
                        "Symbol": symbol,
                        "Company": symbol,
                        "Price": 0.0,
                        "Change": 0.0,
                        "Change%": 0.0,
                        "Open": 0.0,
                        "High": 0.0,
                        "Low": 0.0,
                        "Volume": 0,
                        "Market Cap": 0,
                        "PE Ratio": None,
                        "Sector": "N/A",
                    }
                )

        df = pd.DataFrame(out)
        # ensure consistent dtypes
        numeric_cols = ["Price", "Change", "Change%", "Open", "High", "Low", "Volume", "Market Cap"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_historical_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch historical data using yfinance. Returns DataFrame with Date index."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            hist = hist.reset_index()
            # ensure consistent column names
            if "Date" in hist.columns:
                hist.rename(columns={"Date": "Date"}, inplace=True)
            # cast types
            return hist
        except Exception as e:
            st.warning(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=1800)
    def get_intraday_data(symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data for one trading day."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval=interval)
            hist = hist.reset_index()
            return hist
        except Exception as e:
            st.warning(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()


class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # neutral for early points

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2):
        sma = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std().fillna(0)
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist


# Plot helpers use Plotly but we only import when used to avoid unnecessary startup overhead
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class ChartGenerator:
    @staticmethod
    def create_realtime_overview_chart(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            return fig
        colors = ["green" if cp >= 0 else "red" for cp in df["Change%"].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=df["Symbol"],
                y=df["Price"],
                text=[f"${price:.2f}<br>{change:+.2f}%" for price, change in zip(df["Price"], df["Change%"])],
                textposition="auto",
                marker_color=colors,
                name="Current Price",
            )
        )
        fig.update_layout(
            title="Real-Time Stock Prices Overview",
            xaxis_title="Stock Symbol",
            yaxis_title="Price ($)",
            template="plotly_white",
            showlegend=False,
            height=400,
        )
        return fig

    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, symbol: str, indicators: Dict = None) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} Price Chart", "Volume", "RSI"),
            row_width=[0.6, 0.2, 0.2],
        )
        if df.empty:
            return fig

        xaxis = df["Datetime"] if "Datetime" in df.columns else df["Date"]

        fig.add_trace(
            go.Candlestick(
                x=xaxis, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
            ),
            row=1,
            col=1,
        )

        if indicators:
            if "SMA_20" in indicators:
                fig.add_trace(go.Scatter(x=xaxis, y=indicators["SMA_20"], name="SMA 20"), row=1, col=1)
            if "EMA_12" in indicators:
                fig.add_trace(go.Scatter(x=xaxis, y=indicators["EMA_12"], name="EMA 12"), row=1, col=1)
            if "BB_Upper" in indicators and "BB_Lower" in indicators:
                fig.add_trace(go.Scatter(x=xaxis, y=indicators["BB_Upper"], name="BB Upper", line=dict(dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=xaxis, y=indicators["BB_Lower"], name="BB Lower", line=dict(dash="dash")), row=1, col=1)

        # Volume
        colors = ["green" if c >= o else "red" for c, o in zip(df["Close"].fillna(0), df["Open"].fillna(0))]
        fig.add_trace(go.Bar(x=xaxis, y=df["Volume"], name="Volume", marker_color=colors), row=2, col=1)

        # RSI
        if indicators and "RSI" in indicators:
            fig.add_trace(go.Scatter(x=xaxis, y=indicators["RSI"], name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(template="plotly_white", height=800, showlegend=True, xaxis_rangeslider_visible=False)
        return fig

    @staticmethod
    def create_comparison_chart(symbols: List[str], period: str = "1mo") -> go.Figure:
        fig = go.Figure()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if hist.empty:
                    continue
                normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(x=hist.index, y=normalized, mode="lines", name=symbol, line=dict(width=2)))
            except Exception:
                continue
        fig.update_layout(
            title="Stock Performance Comparison (% Change)",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )
        return fig


# --- Small utilities ---


def format_number(num: float) -> str:
    try:
        num = float(num)
    except Exception:
        return str(num)
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


# --- App main ---
def main():
    st.markdown('<h1 class="main-header">📈 Real-Time Stock Market Dashboard</h1>', unsafe_allow_html=True)

    # session initialization
    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()

    # Sidebar
    st.sidebar.title("⚙️ Dashboard Settings")
    default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_input = st.sidebar.text_input("Enter stock symbols (comma-separated):", value=", ".join(default_stocks))
    selected_stocks = [s.strip().upper() for s in stock_input.split(",") if s.strip()]

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60)

    st.sidebar.subheader("📊 Chart Settings")
    chart_period = st.sidebar.selectbox("Historical Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    show_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)

    if st.sidebar.button("🔄 Refresh Data"):
        # clear caches and force rerun
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.session_state.last_update = datetime.now()
        st.experimental_rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "📈 Detailed Analysis", "🔍 Comparison", "📰 Market Info"])

    with tab1:
        st.subheader("Live Stock Data")
        with st.spinner("Fetching real-time stock data..."):
            realtime_df = StockDataFetcher.get_realtime_data(selected_stocks)

        if not realtime_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_value = realtime_df["Price"].sum()
                st.metric("Total Portfolio Value", format_number(total_value))

            with col2:
                avg_change = realtime_df["Change%"].mean() if "Change%" in realtime_df.columns else 0.0
                st.metric("Average Change %", f"{avg_change:.2f}%")

            with col3:
                gainers = int((realtime_df["Change%"] > 0).sum()) if "Change%" in realtime_df.columns else 0
                st.metric("Gainers", f"{gainers}/{len(realtime_df)}")

            with col4:
                total_volume = int(realtime_df["Volume"].sum()) if "Volume" in realtime_df.columns else 0
                st.metric("Total Volume", format_number(total_volume))

            overview_chart = ChartGenerator.create_realtime_overview_chart(realtime_df)
            st.plotly_chart(overview_chart, use_container_width=True)

            st.subheader("Stock Details")
            display_df = realtime_df.copy()
            # format columns for display (keep numeric types for sorting)
            display_df["PriceStr"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
            display_df["ChangeStr"] = display_df["Change"].apply(lambda x: f"${x:+.2f}")
            display_df["ChangePctStr"] = display_df["Change%"].apply(lambda x: f"{x:+.2f}%")
            display_df["VolumeStr"] = display_df["Volume"].apply(lambda x: f"{int(x):,}")
            display_df["MarketCapStr"] = display_df["Market Cap"].apply(format_number)

            # Build a small dataframe for presentation
            present_df = display_df[
                ["Symbol", "Company", "PriceStr", "ChangeStr", "ChangePctStr", "Open", "High", "Low", "VolumeStr", "MarketCapStr"]
            ].rename(
                columns={
                    "PriceStr": "Price",
                    "ChangeStr": "Change",
                    "ChangePctStr": "Change%",
                    "VolumeStr": "Volume",
                    "MarketCapStr": "Market Cap",
                }
            )

            # styling: apply color based on sign
            def color_change(val):
                try:
                    # value like "+1.23%" or "$+1.23"
                    s = str(val)
                    if s.startswith("+") or s.startswith("$+"):
                        return "color: green"
                    elif s.startswith("-") or s.startswith("$-"):
                        return "color: red"
                except Exception:
                    pass
                return ""

            styled = present_df.style.applymap(color_change, subset=["Change", "Change%"])
            st.dataframe(styled, use_container_width=True)
        else:
            st.error("Unable to fetch stock data. Please check your symbols and try again.")

    with tab2:
        st.subheader("Detailed Technical Analysis")
        selected_symbol = st.selectbox("Select stock for detailed analysis:", selected_stocks or default_stocks)

        if selected_symbol:
            col1, col2 = st.columns([2, 1])
            with col1:
                with st.spinner(f"Loading historical data for {selected_symbol}..."):
                    hist_df = StockDataFetcher.get_historical_data(selected_symbol, chart_period)

                if hist_df.empty or len(hist_df) < 2:
                    st.warning("Not enough historical data to display charts/metrics.")
                else:
                    # safe rename: ensure Date/Datetime present
                    if "Date" in hist_df.columns and "Datetime" not in hist_df.columns:
                        hist_df.rename(columns={"Date": "Date"}, inplace=True)

                    indicators = {}
                    if show_indicators:
                        close = hist_df["Close"]
                        indicators["SMA_20"] = TechnicalIndicators.calculate_sma(close, 20)
                        indicators["EMA_12"] = TechnicalIndicators.calculate_ema(close, 12)
                        indicators["RSI"] = TechnicalIndicators.calculate_rsi(close)
                        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
                        indicators["BB_Upper"], indicators["BB_Middle"], indicators["BB_Lower"] = bb_upper, bb_middle, bb_lower
                        macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(close)
                        indicators["MACD"], indicators["MACD_Signal"], indicators["MACD_HIST"] = macd, macd_signal, macd_hist

                    detailed_chart = ChartGenerator.create_candlestick_chart(hist_df, selected_symbol, indicators)
                    st.plotly_chart(detailed_chart, use_container_width=True)

            with col2:
                if hist_df.empty or len(hist_df) < 2:
                    st.info("No stats available for this symbol.")
                else:
                    st.subheader("Key Statistics")
                    current_price = float(hist_df["Close"].iloc[-1])
                    prev_price = float(hist_df["Close"].iloc[-2])
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0.0
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

                    high_52w = hist_df["High"].max() if "High" in hist_df.columns else current_price
                    low_52w = hist_df["Low"].min() if "Low" in hist_df.columns else current_price
                    avg_volume = int(hist_df["Volume"].mean()) if "Volume" in hist_df.columns else 0

                    st.metric("52W High", f"${high_52w:.2f}")
                    st.metric("52W Low", f"${low_52w:.2f}")
                    st.metric("Avg Volume", f"{avg_volume:,}")

                    if show_indicators and "RSI" in indicators:
                        current_rsi = float(indicators["RSI"].iloc[-1])
                        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                        st.metric("RSI (14)", f"{current_rsi:.2f}", rsi_signal)

    with tab3:
        st.subheader("Stock Performance Comparison")
        comparison_stocks = st.multiselect("Select stocks to compare:", selected_stocks or default_stocks, default=(selected_stocks[:3] if len(selected_stocks) >= 3 else (selected_stocks or default_stocks)))
        if comparison_stocks:
            comparison_period = st.selectbox("Comparison Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2, key="comp_period")
            comparison_chart = ChartGenerator.create_comparison_chart(comparison_stocks, comparison_period)
            st.plotly_chart(comparison_chart, use_container_width=True)

            st.subheader("Performance Summary")
            perf_data = []
            for symbol in comparison_stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=comparison_period)
                    if hist.empty:
                        continue
                    start_price = hist["Close"].iloc[0]
                    end_price = hist["Close"].iloc[-1]
                    total_return = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
                    volatility = hist["Close"].pct_change().std() * np.sqrt(252) * 100 if len(hist) > 1 else 0
                    perf_data.append(
                        {
                            "Symbol": symbol,
                            "Start Price": f"${start_price:.2f}",
                            "End Price": f"${end_price:.2f}",
                            "Total Return": f"{total_return:+.2f}%",
                            "Volatility": f"{volatility:.2f}%",
                        }
                    )
                except Exception:
                    st.warning(f"Could not calculate performance for {symbol}")
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)

    with tab4:
        st.subheader("Market Information & News")
        st.subheader("Major Market Indices")
        indices = {"S&P 500": "SPY", "NASDAQ": "QQQ", "Dow Jones": "DIA", "Russell 2000": "IWM"}
        index_data = []
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if hist.shape[0] >= 2:
                    current = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    change = current - prev
                    change_pct = (change / prev * 100) if prev != 0 else 0.0
                    index_data.append({"Index": name, "Price": f"${current:.2f}", "Change": f"${change:+.2f}", "Change%": f"{change_pct:+.2f}%"})
            except Exception:
                continue
        if index_data:
            index_df = pd.DataFrame(index_data)
            st.dataframe(index_df, use_container_width=True)

        st.subheader("Market Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.info("📈 **Market Status**: Markets are open during regular trading hours (9:30 AM - 4:00 PM ET)")
            st.info("🕐 **Last Updated**: " + st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S"))
        with col2:
            st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.")
            st.success("✅ **Data Sources**: Yahoo Finance (yfinance)")

    # --- Auto-refresh without blocking UI ---
    if auto_refresh:
        now_ts = time.time()
        last_ts = st.session_state.get("last_update_ts", None)
        if last_ts is None:
            st.session_state["last_update_ts"] = now_ts
        # if enough time elapsed, update and rerun (non-blocking)
        if now_ts - st.session_state["last_update_ts"] > refresh_interval:
            st.session_state["last_update_ts"] = now_ts
            st.session_state["last_update"] = datetime.now()
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "📈 Real-Time Stock Market Dashboard | Built with Streamlit & Python | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
