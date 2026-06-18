from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from rich import print

from .data.finviz_client import run_screener, PRESETS
from .data.persistence import create_schema, save_finviz_snapshot, save_recommendations, save_news_articles
from .data.exports import export_table_to_parquet
from .recommend.engine import RuleEngine
from .data.universe import HIGH_INTEREST
from .recommend.options_picker import pick_affordable_contracts
from .backtest.signal_backtest import backtest_short_term
from .broker.factory import make_broker, BROKER_MODE_LEGACY, BROKER_MODE_MCP
from .portfolio.evaluator import evaluate_positions
from .broker.credentials import save_to_keyring, delete_from_keyring, prompt_for_creds
from .execution.policy import ExecutionPolicy, evaluate_order_intent
from .execution.audit import audit_intent, audit_event, audit_policy
from .ops.reporting import (
    RefreshValidationSummary,
    build_db_freshness_report,
    build_report_payload,
    default_json_report_path,
    fetch_api_runtime_status,
    render_textbook_markdown_report,
    write_json_report,
    write_markdown_report,
)
from .ops.maintenance import build_refresh_plan, execute_refresh_plan
from .features.training import build_training_frame


def cmd_init_db(args):
    create_schema(args.db)
    print(f"[green]DB initialized at {args.db}")


def cmd_scan(args):
    failures = []
    saved_rows = 0
    for name in PRESETS.keys():
        print(f"[cyan]Running screener: {name}")
        try:
            df = run_screener(name)
            print(df.head(5))
            save_finviz_snapshot(df, name, db_path=args.db)
            saved_rows += len(df)
        except Exception as e:
            failures.append(name)
            print(f"[red]Scan failed {name}: {e}")
    print(f"[green]Saved snapshots | rows={saved_rows} failed_presets={len(failures)}")


def cmd_recommend(args):
    eng = RuleEngine()
    tickers = HIGH_INTEREST if not args.tickers else [t.strip().upper() for t in args.tickers.split(",")]
    all_recs = []
    for t in tickers:
        try:
            recs = eng.recommend(t)
            all_recs.extend(recs)
            print(f"\n[yellow]{t} recommendations:")
            for r in recs:
                print(f"  - {r.horizon}: {r.bias} conf={r.confidence:.2f} entry={r.entry:.2f} stop={r.stop:.2f} target={r.target:.2f} | {r.notes}")
        except Exception as e:
            print(f"[red]Failed {t}: {e}")
    if args.save:
        save_recommendations(all_recs, db_path=args.db)
        print("[green]Recommendations saved")


def cmd_options(args):
    tickers = HIGH_INTEREST if not args.tickers else [t.strip().upper() for t in args.tickers.split(",")]
    for t in tickers:
        try:
            ideas = pick_affordable_contracts(t, args.horizon, args.bias, args.budget)
            print(f"\n[yellow]{t} {args.horizon} {args.bias} ideas (budget ${args.budget:.0f}):")
            for i in ideas:
                print(f"  - {i.expiry} {i.right} {i.strike:.2f} mid={i.mid:.2f} spread={i.ask - i.bid:.2f} OI={i.open_interest} Vol={i.volume} IV={i.iv:.2f}")
        except Exception as e:
            print(f"[red]Failed {t}: {e}")


def cmd_daily(args):
    print("[cyan]Running daily pipeline: scan -> recommend -> options")
    # ensure schema
    create_schema(args.db)
    # scan all presets
    for name in PRESETS.keys():
        try:
            print(f"[cyan]Screener: {name}")
            df = run_screener(name)
            save_finviz_snapshot(df, name, db_path=args.db)
        except Exception as e:
            print(f"[red]Scan failed {name}: {e}")
    # recommendations for high-interest universe
    eng = RuleEngine()
    all_recs = []
    for t in HIGH_INTEREST:
        try:
            recs = eng.recommend(t)
            all_recs.extend(recs)
        except Exception as e:
            print(f"[red]Rec failed {t}: {e}")
    if all_recs:
        save_recommendations(all_recs, db_path=args.db)
        print(f"[green]Saved {len(all_recs)} recommendations")
    # print affordable 1w/1m longs within budget
    for t in HIGH_INTEREST:
        try:
            ideas = pick_affordable_contracts(t, "1w", "long", budget=args.budget)
            if ideas:
                print(f"\n[yellow]{t} 1w long ideas (budget ${args.budget:.0f}):")
                for i in ideas:
                    print(f"  - {i.expiry} {i.right} {i.strike:.2f} mid={i.mid:.2f} OI={i.open_interest} Vol={i.volume} IV={i.iv:.2f}")
        except Exception as e:
            print(f"[red]Options failed {t}: {e}")


def cmd_backtest(args):
    tickers = HIGH_INTEREST if not args.tickers else [t.strip().upper() for t in args.tickers.split(",")]
    for t in tickers:
        try:
            rpt = backtest_short_term(
                t,
                start=args.start,
                entry_rsi_threshold=args.entry_rsi_threshold,
                max_hold_days=args.max_hold_days,
                stop_loss_pct=args.stop_loss_pct,
                ma_filter=args.ma_filter,
                take_profit_pct=args.take_profit_pct,
                ma_trend_filter=args.ma_trend_filter,
            )
            print(f"\n[yellow]{t} short-term backtest ({args.start}+):")
            print(
                f"  params: rsi>{args.entry_rsi_threshold:.2f} "
                f"maxHold={args.max_hold_days} stopLoss={args.stop_loss_pct:.2f}% "
                f"takeProfit={args.take_profit_pct:.2f}% ma={args.ma_filter} trend={args.ma_trend_filter}"
            )
            print(f"  trades={rpt.n_trades} win%={rpt.win_rate:.2%} avgRet={rpt.avg_ret:.2%} medRet={rpt.median_ret:.2%}")
            print(f"  hold={rpt.avg_hold_days:.1f}d maxDD={rpt.max_dd:.2%} Sharpe={rpt.sharpe:.2f} Sortino={rpt.sortino:.2f} CAGR={rpt.cagr:.2%} PF={rpt.profit_factor:.2f}")
        except Exception as e:
            print(f"[red]Backtest failed {t}: {e}")


def cmd_robinhood_login(args):
    creds = prompt_for_creds()
    try:
        save_to_keyring(creds.username, creds.password)
        print("[green]Saved Robinhood credentials to OS keyring")
    except Exception as e:
        print(f"[red]Failed to save to keyring: {e}")


def cmd_robinhood_logout(args):
    delete_from_keyring()
    print("[green]Deleted Robinhood credentials from OS keyring")


def cmd_portfolio(args):
    broker = make_broker(args.broker_mode)
    try:
        sigs = evaluate_positions(broker)
        print("[cyan]Position signals:")
        for s in sigs:
            print(f"  - {s.ticker} [{s.side}] -> {s.action} ({s.horizon}) conf={s.confidence:.2f} | {s.notes}")
    except Exception as e:
        print(f"[red]Portfolio evaluation failed: {e}")


def cmd_propose_order(args):
    """Create a policy-evaluated order intent and save audit records.

    This does not place a broker order.
    """
    create_schema(args.db)
    policy = ExecutionPolicy(
        require_approval=(not args.auto),
        max_daily_notional=args.max_daily_notional,
        max_orders_per_day=args.max_orders_per_day,
        max_position_notional=args.max_position_notional,
    )
    decision = evaluate_order_intent(
        notional=args.notional,
        orders_today=args.orders_today,
        position_notional_after=args.position_notional_after,
        policy=policy,
    )

    intent_id = audit_intent(
        ticker=args.ticker.upper(),
        action=args.action,
        qty=args.qty,
        notional=args.notional,
        broker_mode=args.broker_mode,
        status="approved" if decision.allow else "blocked",
        db_path=args.db,
    )
    audit_policy(intent_id=intent_id, allow=decision.allow, reason=decision.reason, db_path=args.db)
    audit_event(intent_id=intent_id, event_type="policy_check", message=decision.reason, db_path=args.db)

    verdict = "APPROVED" if decision.allow else "BLOCKED"
    print(f"[yellow]Intent {intent_id}: {verdict} | {decision.reason}")


def _run_scan_pipeline_for_refresh(db_path: str) -> tuple[int, int, list[str]]:
    total_saved = 0
    successes = 0
    failures = []
    for name in PRESETS.keys():
        print(f"[cyan]Refreshing screener snapshot: {name}")
        try:
            df = run_screener(name)
            save_finviz_snapshot(df, name, db_path=db_path)
            total_saved += len(df)
            successes += 1
        except Exception as e:
            failures.append(name)
            print(f"[red]Snapshot refresh failed {name}: {e}")

    print(
        "[yellow]Snapshot refresh summary | "
        f"successful_presets={successes} failed_presets={len(failures)}"
    )

    if successes == 0:
        raise RuntimeError(
            "Snapshot refresh failed for all presets; check Finviz access/proxy settings."
        )

    return total_saved, successes, failures


def _run_recommend_pipeline_for_refresh(db_path: str) -> tuple[int, int, list[str]]:
    eng = RuleEngine()
    all_recs = []
    success_count = 0
    failures = []
    for t in HIGH_INTEREST:
        try:
            all_recs.extend(eng.recommend(t))
            success_count += 1
        except Exception as e:
            failures.append(t)
            print(f"[red]Recommendation refresh failed {t}: {e}")
    if all_recs:
        save_recommendations(all_recs, db_path=db_path)
    return len(all_recs), success_count, failures


def cmd_ops_refresh(args):
    create_schema(args.db)
    report = build_db_freshness_report(args.db)
    plan = build_refresh_plan(report, stale_days=args.stale_days, force=args.force)

    print(
        "[yellow]Refresh decision | "
        f"snapshots={plan.refresh_snapshots} "
        f"recommendations={plan.refresh_recommendations}"
    )

    validation = RefreshValidationSummary(
        snapshot_rows_saved=0,
        recommendations_saved=0,
        successful_presets=0,
        failed_presets=[],
        successful_recommendation_tickers=0,
        failed_recommendation_tickers=[],
    )

    def _scan_wrapper() -> int:
        rows, successes, failures = _run_scan_pipeline_for_refresh(args.db)
        validation.snapshot_rows_saved = rows
        validation.successful_presets = successes
        validation.failed_presets = failures
        return rows

    def _recommend_wrapper() -> int:
        saved, success_count, failures = _run_recommend_pipeline_for_refresh(args.db)
        validation.recommendations_saved = saved
        validation.successful_recommendation_tickers = success_count
        validation.failed_recommendation_tickers = failures
        return saved

    result = execute_refresh_plan(
        plan,
        run_scan=_scan_wrapper,
        run_recommend=_recommend_wrapper,
    )

    print(
        "[green]Refresh completed | "
        f"snapshot_rows_saved={result.snapshots_rows_saved} "
        f"recommendations_saved={result.recommendations_saved}"
    )

    if args.with_report:
        runtime = fetch_api_runtime_status(args.api_base) if args.include_api else None
        post_report = build_db_freshness_report(args.db)
        markdown = render_textbook_markdown_report(
            post_report,
            runtime=runtime,
            stale_days=args.stale_days,
            validation=validation,
        )
        payload = build_report_payload(
            post_report,
            runtime=runtime,
            stale_days=args.stale_days,
            validation=validation,
        )
        out_path = write_markdown_report(markdown, args.report_out)
        json_path = write_json_report(payload, default_json_report_path(args.report_out))
        print(f"[green]Wrote report: {out_path}")
        print(f"[green]Wrote json report: {json_path}")


def cmd_ops_report(args):
    create_schema(args.db)
    report = build_db_freshness_report(args.db)
    runtime = fetch_api_runtime_status(args.api_base) if args.include_api else None
    markdown = render_textbook_markdown_report(report, runtime=runtime, stale_days=args.stale_days)
    payload = build_report_payload(report, runtime=runtime, stale_days=args.stale_days)
    out_path = write_markdown_report(markdown, args.report_out)
    json_path = write_json_report(payload, default_json_report_path(args.report_out))
    print(f"[green]Wrote report: {out_path}")
    print(f"[green]Wrote json report: {json_path}")


def _load_article_rows(input_path: str) -> list[dict]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input archive file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "items" in payload and isinstance(payload["items"], list):
                return payload["items"]
            return [payload]
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON archive must be a list or object payload")

    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if suffix in {".csv"}:
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")

    raise ValueError("Supported archive formats: .json, .jsonl, .ndjson, .csv")


def cmd_train_features(args):
    create_schema(args.db)
    tickers = HIGH_INTEREST if not args.tickers else [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    frame = build_training_frame(
        tickers=tickers,
        db_path=args.db,
        period=args.period,
        interval=args.interval,
        lookback_days=args.lookback_days,
        forecast_horizon=args.forecast_horizon,
    )
    print(frame.head(10))
    if args.save:
        from .data.persistence import save_training_features

        batch_id = save_training_features(frame.to_dict(orient="records"), db_path=args.db)
        print(f"[green]Saved training features batch: {batch_id}")
    if args.out:
        out_path = export_table_to_parquet("training_feature_snapshots", args.db, args.out)
        print(f"[green]Exported training features parquet: {out_path}")


def cmd_news_ingest(args):
    create_schema(args.db)
    rows = _load_article_rows(args.input)
    saved = save_news_articles(rows, db_path=args.db)
    print(f"[green]Saved news articles: {saved}")
    if args.out:
        out_path = export_table_to_parquet("news_articles", args.db, args.out)
        print(f"[green]Exported news archive parquet: {out_path}")


def cmd_export_table(args):
    out_path = export_table_to_parquet(args.table, args.db, args.out)
    print(f"[green]Exported {args.table} to parquet: {out_path}")


def main():
    p = argparse.ArgumentParser(prog="quantflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("init-db")
    p1.add_argument("--db", default="sqlite:///quantflow.db")
    p1.set_defaults(func=cmd_init_db)

    p2 = sub.add_parser("scan")
    p2.add_argument("--db", default="sqlite:///quantflow.db")
    p2.set_defaults(func=cmd_scan)

    p3 = sub.add_parser("rec")
    p3.add_argument("--tickers", default="")
    p3.add_argument("--save", action="store_true")
    p3.add_argument("--db", default="sqlite:///quantflow.db")
    p3.set_defaults(func=cmd_recommend)

    p4 = sub.add_parser("options")
    p4.add_argument("--tickers", default="")
    p4.add_argument("--horizon", default="1m", choices=["1w","1m","3m","6m","1y"]) 
    p4.add_argument("--bias", default="long", choices=["long","short"]) 
    p4.add_argument("--budget", type=float, default=300.0)
    p4.set_defaults(func=cmd_options)

    p5 = sub.add_parser("daily")
    p5.add_argument("--db", default="sqlite:///quantflow.db")
    p5.add_argument("--budget", type=float, default=300.0)
    p5.set_defaults(func=cmd_daily)

    p6 = sub.add_parser("backtest")
    p6.add_argument("--tickers", default="")
    p6.add_argument("--start", default="2018-01-01")
    p6.add_argument("--entry-rsi-threshold", type=float, default=50.0)
    p6.add_argument("--max-hold-days", type=int, default=7)
    p6.add_argument("--stop-loss-pct", type=float, default=0.0)
    p6.add_argument("--ma-filter", default="sma20", choices=["sma20", "sma50", "sma200"])
    p6.add_argument("--take-profit-pct", type=float, default=0.0)
    p6.add_argument("--ma-trend-filter", default="none", choices=["none", "above_sma50", "above_sma200"])
    p6.set_defaults(func=cmd_backtest)

    p7 = sub.add_parser("portfolio")
    p7.add_argument("--broker-mode", default=BROKER_MODE_LEGACY, choices=[BROKER_MODE_LEGACY, BROKER_MODE_MCP])
    p7.set_defaults(func=cmd_portfolio)

    p8 = sub.add_parser("robinhood-login")
    p8.set_defaults(func=cmd_robinhood_login)

    p9 = sub.add_parser("robinhood-logout")
    p9.set_defaults(func=cmd_robinhood_logout)

    p10 = sub.add_parser("propose-order")
    p10.add_argument("--ticker", required=True)
    p10.add_argument("--action", default="buy", choices=["buy", "sell", "close", "rebalance"])
    p10.add_argument("--qty", type=float, default=1.0)
    p10.add_argument("--notional", type=float, required=True)
    p10.add_argument("--orders-today", type=int, default=0)
    p10.add_argument("--position-notional-after", type=float, default=0.0)
    p10.add_argument("--auto", action="store_true", help="Allow pass-through if policy checks pass.")
    p10.add_argument("--max-daily-notional", type=float, default=5000.0)
    p10.add_argument("--max-orders-per-day", type=int, default=20)
    p10.add_argument("--max-position-notional", type=float, default=2000.0)
    p10.add_argument("--broker-mode", default=BROKER_MODE_LEGACY, choices=[BROKER_MODE_LEGACY, BROKER_MODE_MCP])
    p10.add_argument("--db", default="sqlite:///quantflow.db")
    p10.set_defaults(func=cmd_propose_order)

    p11 = sub.add_parser("ops-refresh")
    p11.add_argument("--db", default="sqlite:///quantflow.db")
    p11.add_argument("--stale-days", type=int, default=7)
    p11.add_argument("--force", action="store_true")
    p11.add_argument("--with-report", action="store_true")
    p11.add_argument("--include-api", action="store_true")
    p11.add_argument("--api-base", default="http://127.0.0.1:8100")
    p11.add_argument("--report-out", default="docs/LOCAL_SYSTEM_REPORT.md")
    p11.set_defaults(func=cmd_ops_refresh)

    p12 = sub.add_parser("ops-report")
    p12.add_argument("--db", default="sqlite:///quantflow.db")
    p12.add_argument("--stale-days", type=int, default=7)
    p12.add_argument("--include-api", action="store_true")
    p12.add_argument("--api-base", default="http://127.0.0.1:8100")
    p12.add_argument("--report-out", default="docs/LOCAL_SYSTEM_REPORT.md")
    p12.set_defaults(func=cmd_ops_report)

    p13 = sub.add_parser("train-features")
    p13.add_argument("--tickers", default="")
    p13.add_argument("--db", default="sqlite:///quantflow.db")
    p13.add_argument("--period", default="2y")
    p13.add_argument("--interval", default="1d")
    p13.add_argument("--lookback-days", type=int, default=30)
    p13.add_argument("--forecast-horizon", type=int, default=20)
    p13.add_argument("--save", action="store_true")
    p13.add_argument("--out", default="")
    p13.set_defaults(func=cmd_train_features)

    p14 = sub.add_parser("news-ingest")
    p14.add_argument("--input", required=True)
    p14.add_argument("--db", default="sqlite:///quantflow.db")
    p14.add_argument("--out", default="")
    p14.set_defaults(func=cmd_news_ingest)

    p15 = sub.add_parser("export-table")
    p15.add_argument("--table", required=True)
    p15.add_argument("--db", default="sqlite:///quantflow.db")
    p15.add_argument("--out", required=True)
    p15.set_defaults(func=cmd_export_table)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
