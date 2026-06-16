from __future__ import annotations

import argparse
from datetime import datetime
from rich import print

from .data.finviz_client import run_screener, PRESETS
from .data.persistence import create_schema, save_finviz_snapshot, save_recommendations
from .recommend.engine import RuleEngine
from .data.universe import HIGH_INTEREST
from .recommend.options_picker import pick_affordable_contracts
from .backtest.signal_backtest import backtest_short_term
from .broker.factory import make_broker, BROKER_MODE_LEGACY, BROKER_MODE_MCP
from .portfolio.evaluator import evaluate_positions
from .broker.credentials import save_to_keyring, delete_from_keyring, prompt_for_creds
from .execution.policy import ExecutionPolicy, evaluate_order_intent
from .execution.audit import audit_intent, audit_event, audit_policy


def cmd_init_db(args):
    create_schema(args.db)
    print(f"[green]DB initialized at {args.db}")


def cmd_scan(args):
    for name in PRESETS.keys():
        print(f"[cyan]Running screener: {name}")
        df = run_screener(name)
        print(df.head(5))
        save_finviz_snapshot(df, name, db_path=args.db)
    print("[green]Saved snapshots")


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
            rpt = backtest_short_term(t, start=args.start)
            print(f"\n[yellow]{t} short-term backtest ({args.start}+):")
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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
