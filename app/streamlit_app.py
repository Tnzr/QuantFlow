import streamlit as st
import pandas as pd
import plotly.express as px
import os
import concurrent.futures

from quantflow.data.queries import latest_universe, latest_recommendations_per_ticker
from quantflow.data.universe import HIGH_INTEREST
from quantflow.recommend.options_picker import pick_affordable_contracts
from quantflow.recommend.engine import RuleEngine
from quantflow.data.finviz_client import PRESETS, run_screener
from quantflow.data.persistence import create_schema, save_finviz_snapshot, save_recommendations
from quantflow.features.indicators import fetch_ohlcv, compute_indicators
from quantflow.features.seasonality import seasonality_by_doy, proximity_to_earnings
from quantflow.broker.robinhood import RobinhoodBroker
from quantflow.portfolio.evaluator import evaluate_positions
from quantflow.backtest.signal_backtest import backtest_short_term

st.set_page_config(page_title="QuantFlow", layout="wide")

st.title("QuantFlow PoC Dashboard")

with st.sidebar.expander("Help"):
    st.write("Run 'make daily' or use Run Pipeline page to populate the DB, then use this dashboard.")
    st.write("Options Ideas uses budget, horizon, and delta-aware heuristics.")

os.makedirs(os.path.join(os.path.dirname(__file__), '../data/database'), exist_ok=True)
def_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/database/quantflow.db'))
DB = st.sidebar.text_input("DB URL", value=f"sqlite:///{def_db_path}")

page = st.sidebar.radio(
    "View",
    [
        "Scanner & Recs",
        "Options Ideas",
        "Run Pipeline",
        "Portfolio",
        "Charts",
        "Backtest",
    ],
    index=0,
)

if page == "Scanner & Recs":
    st.subheader("Latest universe snapshot")
    try:
        dfu = latest_universe(db_path=DB)
        if dfu is None or dfu.empty:
            st.info("No data yet. Run 'make scan' or use Run Pipeline.")
        else:
            st.dataframe(dfu)
    except Exception as e:
        st.error(f"Failed loading universe: {e}")

    st.subheader("Latest recommendations per ticker")
    try:
        dfr = latest_recommendations_per_ticker(db_path=DB)
        if dfr is None or dfr.empty:
            st.info("No recommendations saved yet. Run recommender or pipeline.")
        else:
            st.dataframe(dfr)
            st.download_button(
                "Download CSV",
                data=dfr.to_csv(index=False).encode("utf-8"),
                file_name="latest_recommendations.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed loading recommendations: {e}")

elif page == "Options Ideas":
    st.subheader("Affordable option ideas")
    tickers = st.multiselect("Tickers", options=HIGH_INTEREST, default=["AAPL","MSFT","NVDA"]) 
    horizon = st.selectbox("Horizon", ["1w","1m","3m","6m","1y"], index=1)
    bias = st.selectbox("Bias", ["long","short"], index=0)
    budget = st.number_input("Max premium ($)", value=300.0, min_value=20.0, step=10.0)
    if st.button("Find candidates"):
        with st.spinner("Scanning option chains..."):
            rows = []
            for t in tickers:
                try:
                    ideas = pick_affordable_contracts(t, horizon=horizon, bias=bias, budget=float(budget))
                    for i in ideas:
                        rows.append({
                            "ticker": i.ticker,
                            "horizon": i.horizon,
                            "bias": i.bias,
                            "expiry": i.expiry,
                            "right": i.right,
                            "strike": i.strike,
                            "bid": i.bid,
                            "ask": i.ask,
                            "mid": i.mid,
                            "volume": i.volume,
                            "open_interest": i.open_interest,
                            "iv": i.iv,
                            "delta_band": i.delta_band,
                            "rationale": i.rationale,
                        })
                except Exception as e:
                    st.warning(f"{t}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="option_ideas.csv", mime="text/csv")
        else:
            st.info("No candidates matched filters.")

elif page == "Run Pipeline":
    st.subheader("Run screener and recommender")
    st.write("This runs Finviz presets and saves snapshots, then generates recommendations for a default universe.")
    preset_names = list(PRESETS.keys())
    if preset_names:
        st.caption(f"Loaded presets ({len(preset_names)}): " + ", ".join(preset_names))
    else:
        st.warning("No Finviz presets loaded. Check configs/finviz_presets.yml and finviz client setup.")
    # User option: parallel or sequential
    parallel = st.checkbox("Run screeners in parallel (faster)", value=True)
    max_workers = st.slider("Parallel workers", min_value=1, max_value=8, value=4)
    if st.button("Run Scan Presets"):
        with st.spinner("Running Finviz screeners and saving snapshots..."):
            try:
                create_schema(DB)
                if parallel:
                    from quantflow.data.finviz_client import run_screeners_parallel
                    results = run_screeners_parallel(preset_names, max_workers=max_workers)
                    for name, df in results.items():
                        if df is not None:
                            save_finviz_snapshot(df, name, db_path=DB)
                else:
                    for name in preset_names:
                        df = run_screener(name)
                        save_finviz_snapshot(df, name, db_path=DB)
                st.success("Scans saved.")
            except Exception as e:
                st.error(f"Scan failed: {e}")
    if st.button("Run Recommendations (save)"):
        with st.spinner("Computing recommendations..."):
            try:
                eng = RuleEngine()
                all_recs = []
                for t in HIGH_INTEREST:
                    try:
                        recs = eng.recommend(t)
                        all_recs.extend(recs)
                    except Exception as e:
                        st.warning(f"{t}: {e}")
                if all_recs:
                    save_recommendations(all_recs, db_path=DB)
                    st.success(f"Saved {len(all_recs)} recommendations.")
                else:
                    st.info("No recommendations generated.")
            except Exception as e:
                st.error(f"Recommend failed: {e}")

elif page == "Portfolio":
    st.subheader("Robinhood portfolio (read-only)")
    st.caption("Credentials are used for this session only (not stored).")
    with st.expander("Login", expanded=not st.session_state.get("rh_logged_in", False)):
        with st.form("rh_login"):
            u = st.text_input("Username (email)")
            p = st.text_input("Password", type="password")
            mfa = st.text_input("MFA Code (optional)")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not u or not p:
                    st.error("Enter username and password.")
                else:
                    os.environ["ROBINHOOD_USERNAME"] = u
                    os.environ["ROBINHOOD_PASSWORD"] = p
                    if mfa:
                        os.environ["ROBINHOOD_MFA"] = mfa
                    try:
                        broker = RobinhoodBroker()
                        broker.login()
                        st.session_state["rh_logged_in"] = True
                        st.success("Logged in.")
                    except Exception as e:
                        st.error(f"Login failed: {e}")
    if st.session_state.get("rh_logged_in"):
        broker = RobinhoodBroker()
        try:
            sigs = evaluate_positions(broker)
            if not sigs:
                st.info("No positions or no signals.")
            else:
                rows = [s.__dict__ for s in sigs]
                df = pd.DataFrame(rows)
                # Add color-coded action tags
                def tag(a: str) -> str:
                    color = {"exit": "red", "add": "green", "hold": "gray"}.get(a, "blue")
                    return f"<span style='color:{color};font-weight:600'>{a.upper()}</span>"
                df_display = df.copy()
                df_display["action"] = df_display["action"].apply(tag)
                st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="position_signals.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Portfolio evaluation failed: {e}")

elif page == "Charts":
    st.subheader("Seasonality and indicators")
    t = st.selectbox("Ticker", options=HIGH_INTEREST, index=0)
    if st.button("Load charts"):
        with st.spinner("Loading OHLCV and computing indicators..."):
            try:
                df = fetch_ohlcv(t, period="6m")
                df = compute_indicators(df)
                fig = px.line(df.reset_index(), x="date", y=["adj close","sma20","sma50","sma200"], title=f"{t} Price & SMAs")
                st.plotly_chart(fig, use_container_width=True)
                rsi_fig = px.line(df.reset_index(), x="date", y="rsi14", title=f"{t} RSI14")
                st.plotly_chart(rsi_fig, use_container_width=True)
                doy = seasonality_by_doy(t, years=10)
                seas_fig = px.line(doy, x="doy", y="avg_ret", title=f"{t} DOY Seasonality (avg daily return)")
                st.plotly_chart(seas_fig, use_container_width=True)
                days = proximity_to_earnings(t)
                st.info(f"Earnings in: {days} days" if days is not None else "Earnings date: unknown")
            except Exception as e:
                st.error(f"Chart load failed: {e}")

elif page == "Backtest":
    st.subheader("Short-term backtest")
    t = st.selectbox("Ticker", options=HIGH_INTEREST, index=0, key="bt_t")
    start = st.text_input("Start date (YYYY-MM-DD)", value="2018-01-01")
    entry_rsi_threshold = st.number_input("Entry RSI threshold", value=50.0, step=1.0)
    max_hold_days = st.number_input("Max hold days", min_value=1, value=7, step=1)
    stop_loss_pct = st.number_input("Stop loss % (0 disables)", min_value=0.0, value=0.0, step=0.5)
    take_profit_pct = st.number_input("Take profit % (0 disables)", min_value=0.0, value=0.0, step=0.5)
    ma_filter = st.selectbox("MA filter", options=["sma20", "sma50", "sma200"], index=0)
    ma_trend_filter = st.selectbox("Entry trend filter", options=["none", "above_sma50", "above_sma200"], index=0)
    if st.button("Run backtest"):
        with st.spinner("Running backtest..."):
            try:
                rpt = backtest_short_term(
                    ticker=t,
                    start=start,
                    entry_rsi_threshold=float(entry_rsi_threshold),
                    max_hold_days=int(max_hold_days),
                    stop_loss_pct=float(stop_loss_pct),
                    ma_filter=ma_filter,
                    take_profit_pct=float(take_profit_pct),
                    ma_trend_filter=ma_trend_filter,
                )
                st.write(f"Trades: {rpt.n_trades}")
                st.write(f"Win rate: {rpt.win_rate:.2%}")
                st.write(f"Avg ret: {rpt.avg_ret:.2%} | Median ret: {rpt.median_ret:.2%}")
                st.write(f"Hold: {rpt.avg_hold_days:.1f}d | MaxDD: {rpt.max_dd:.2%}")
                st.write(f"Sharpe: {rpt.sharpe:.2f} | Sortino: {rpt.sortino:.2f} | CAGR: {rpt.cagr:.2%} | PF: {rpt.profit_factor:.2f}")
                eq = rpt.equity_curve.reset_index().rename(columns={"date":"Date", 0:"equity"})
                if "equity" not in eq.columns:
                    eq = eq.rename(columns={eq.columns[-1]: "equity"})
                fig = px.line(eq, x="Date", y="equity", title=f"{t} Equity Curve")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
