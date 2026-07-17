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
from .broker.factory import make_broker, BROKER_MODE_MCP
from .portfolio.evaluator import evaluate_positions
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


def _lazy_import_models():
    from .models.dataset import build_dataset as _build_dataset
    from .models.trainer import train_model, TrainingConfig, Trainer
    from .models.losses import LossConfig
    from .models.backtest_engine import AIBacktestEngine
    from .models.architectures import create_model
    return _build_dataset, train_model, TrainingConfig, Trainer, LossConfig, AIBacktestEngine, create_model


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


def cmd_build_dataset(args):
    from .data.universe import SECTOR_UNIVERSE
    build_model_dataset, _, _, _, _, _, _ = _lazy_import_models()

    print("[cyan]Building ML training dataset...")
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.sector:
        tickers = SECTOR_UNIVERSE.get(args.sector, []) + SECTOR_UNIVERSE.get(f"{args.sector} — Alternatives", [])
    if not tickers:
        tickers = SECTOR_UNIVERSE.get("Financials", []) + SECTOR_UNIVERSE.get("Financials — Alternatives", [])

    print(f"[yellow]Tickers: {tickers}")
    df = build_model_dataset(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        snapshot_step=args.snapshot_step,
        max_rows=args.max_rows or None,
        export_path=args.out or None,
    )
    print(f"[green]Dataset built: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"[green]Columns: {list(df.columns)}")

    if "event_state_code" in df.columns:
        dist = df["event_state_code"].value_counts().to_dict()
        for code in sorted(dist):
            label = {0: "inter_event", 1: "pre_event", 2: "onset"}.get(code, str(code))
            print(f"  {label}: {dist[code]} ({100*dist[code]/len(df):.1f}%)")


def cmd_train(args):
    from quantflow.data.labeler import build_labeled_dataset
    from .data.universe import SECTOR_UNIVERSE
    _, train_model, TrainingConfig, Trainer, LossConfig, _, _ = _lazy_import_models()

    print("[cyan]Loading dataset...")
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    df = build_labeled_dataset(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        snapshot_step=args.snapshot_step,
        forecast_horizon=args.forecast_horizon,
        max_rows=args.max_rows or None,
    )

    print(f"[yellow]Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        architecture=args.arch,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_multi_scale=not args.no_multi_scale,
        use_coherent_heads=not args.no_coherent_heads,
        checkpoint_dir=args.checkpoint_dir,
    )

    loss_config = LossConfig(
        classification_weight=args.cls_weight,
        regression_weight=args.reg_weight,
        coherence_weight=args.coh_weight,
    )

    print(f"[cyan]Training {args.arch} model (hidden_dim={args.hidden_dim}, epochs={args.epochs})...")
    trainer, history, predictions = train_model(df, config=config, loss_config=loss_config, device=args.device)

    if args.save:
        trainer.save_model(args.save)
    print(f"[green]Training complete. Best val loss: {trainer.best_val_loss:.4f}")


def cmd_ai_backtest(args):
    print("[cyan]AI Backtest — loading model and running backtest...")
    import torch
    from .data.universe import SECTOR_UNIVERSE
    _, _, TrainingConfig, Trainer, _, AIBacktestEngine, create_model = _lazy_import_models()
    from .models.dataset import FEATURE_COLUMNS

    if not args.model:
        print("[red]Error: --model path is required")
        return

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else (
        SECTOR_UNIVERSE.get("Financials", []) + SECTOR_UNIVERSE.get("Financials — Alternatives", [])
    )[:5]

    print(f"[yellow]Tickers: {tickers}")

    config = TrainingConfig(
        architecture=args.arch,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        hidden_dim=args.hidden_dim,
    )

    from .models.dataset import FEATURE_COLUMNS
    model = create_model(
        architecture=config.architecture,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=config.hidden_dim,
        forecast_horizon=config.forecast_horizon,
        use_multi_scale=not args.no_multi_scale,
        use_coherent_heads=not args.no_coherent_heads,
    )

    trainer = Trainer(model, config=config, device=args.device)
    trainer.load_model(args.model)

    engine = AIBacktestEngine(
        trainer=trainer,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        max_hold_days=args.max_hold_days,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
    )

    result = engine.run(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        start_date=args.start,
    )

    print(f"\n[yellow]Backtest Results:")
    print(f"  Trades: {result.n_trades}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Avg Return: {result.avg_ret:.2%}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Sortino: {result.sortino_ratio:.2f}")
    print(f"  CAGR: {result.cagr:.2%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Total Return: {result.total_return:.2%}")

    if not result.by_ticker.empty:
        print(f"\n[yellow]By Ticker:")
        for row in result.by_ticker.to_dict(orient="records"):
            print(f"  {row['ticker']}: trades={row['n_trades']} win={row['win_rate']:.2%} avg_ret={row['avg_ret']:.2%}")

    if args.viz:
        try:
            from .models.visualization import generate_backtest_visualizations
            import pandas as pd

            print(f"\n[cyan]Generating visualizations from portfolio signals...")

            signals = {}
            daily_signals = getattr(result, "daily_signals", {})
            if daily_signals:
                first_tk = next(iter(daily_signals.keys()), None)
                if first_tk and not daily_signals[first_tk].empty:
                    df_sig = daily_signals[first_tk]
                    signals = {
                        "prob_inter_event": df_sig.get("prob_inter_event", pd.Series(dtype=float)).values,
                        "prob_pre_event": df_sig.get("prob_pre_event", pd.Series(dtype=float)).values,
                        "prob_onset": df_sig.get("prob_onset", pd.Series(dtype=float)).values,
                        "forecast_tau": df_sig.get("forecast_tau", pd.Series(dtype=float)).values,
                    }

            paths = generate_backtest_visualizations(
                result, signals,
                output_dir=args.viz_dir,
                ticker="portfolio",
                prefix="ai_backtest",
            )
            for name, path in paths.items():
                if path:
                    print(f"  [green]{name}[/green]: {path}")
        except Exception as e:
            print(f"[red]Visualization failed: {e}")


def cmd_predict(args):
    print("[cyan]AI Prediction — loading model and running inference...")
    import torch
    import pandas as pd
    import numpy as np
    from .data.universe import HIGH_INTEREST
    from .features.indicators import fetch_ohlcv, compute_indicators
    from .models.dataset import FEATURE_COLUMNS, _prepare_features
    _, _, _, _, _, _, create_model = _lazy_import_models()

    if not args.model:
        print("[red]Error: --model path is required")
        return

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else HIGH_INTEREST[:10]

    config = TrainingConfig(
        architecture=args.arch,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
    )

    model = create_model(
        architecture=config.architecture,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=config.hidden_dim,
        forecast_horizon=args.forecast_horizon,
        use_multi_scale=not args.no_multi_scale,
        use_coherent_heads=not args.no_coherent_heads,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    results = []
    for ticker in tickers:
        try:
            from .data.labeler import _build_features_at
            from .models.dataset import FEATURE_COLUMNS

            df = fetch_ohlcv(ticker, period=args.period)
            df = compute_indicators(df)
            if len(df) < args.lookback:
                results.append({"ticker": ticker, "error": "insufficient data"})
                continue

            feature_rows = []
            start_idx = max(0, len(df) - args.lookback * 2)
            for idx in range(start_idx, len(df)):
                feat = _build_features_at(df, idx, ticker=ticker, forecast_horizon=args.forecast_horizon)
                if "error" not in feat:
                    feature_rows.append(feat)

            if len(feature_rows) < args.lookback:
                results.append({"ticker": ticker, "error": f"insufficient features: {len(feature_rows)}"})
                continue

            feat_df = pd.DataFrame(feature_rows)
            feat_df = _prepare_features(feat_df)

            available_cols = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
            features = feat_df[available_cols].values[-args.lookback:].astype(np.float32)

            if features.shape[0] < args.lookback:
                results.append({"ticker": ticker, "error": "insufficient lookback"})
                continue

            if features.shape[1] != len(FEATURE_COLUMNS):
                if features.shape[1] > len(FEATURE_COLUMNS):
                    features = features[:, :len(FEATURE_COLUMNS)]
                else:
                    pad = np.zeros((features.shape[0], len(FEATURE_COLUMNS) - features.shape[1]), dtype=np.float32)
                    features = np.concatenate([features, pad], axis=1)

            features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(features_clean).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(x)

            probs = outputs["past_state_probs"].cpu().numpy()[0]
            forecast = float(outputs["future_forecast"].cpu().numpy()[0].mean())
            sigma = float(outputs["uncertainty_sigma"].cpu().numpy()[0].mean())

            results.append({
                "ticker": ticker,
                "price": float(df["adj close"].iloc[-1]),
                "state": int(np.argmax(probs)),
                "state_label": {0: "inter_event", 1: "pre_event", 2: "onset"}.get(int(np.argmax(probs)), "unknown"),
                "prob_inter_event": float(probs[0]),
                "prob_pre_event": float(probs[1]),
                "prob_onset": float(probs[2]),
                "forecast_tau_days": forecast,
                "uncertainty": sigma,
            })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    print(f"\n[yellow]AI Predictions ({len(results)} tickers)")
    for r in sorted(results, key=lambda x: x.get("prob_pre_event", 0) + x.get("prob_onset", 0) * 1.5, reverse=True):
        if "error" in r:
            print(f"  {r['ticker']}: [red]ERROR — {r['error']}")
        else:
            state_color = "green" if r["state"] == 0 else ("yellow" if r["state"] == 1 else "red")
            print(
                f"  [{state_color}]{r['state_label']:>12}[/{state_color}] "
                f"{r['ticker']:>6} "
                f"${r['price']:.2f} "
                f"inter={r['prob_inter_event']:.2f} pre={r['prob_pre_event']:.2f} onset={r['prob_onset']:.2f} "
                f"tau={r['forecast_tau_days']:.1f}d unc={r['uncertainty']:.3f}"
            )


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
    p7.add_argument("--broker-mode", default=BROKER_MODE_MCP, choices=[BROKER_MODE_MCP])
    p7.set_defaults(func=cmd_portfolio)

    p8 = sub.add_parser("propose-order")
    p8.add_argument("--ticker", required=True)
    p8.add_argument("--action", default="buy", choices=["buy", "sell", "close", "rebalance"])
    p8.add_argument("--qty", type=float, default=1.0)
    p8.add_argument("--notional", type=float, required=True)
    p8.add_argument("--orders-today", type=int, default=0)
    p8.add_argument("--position-notional-after", type=float, default=0.0)
    p8.add_argument("--auto", action="store_true", help="Allow pass-through if policy checks pass.")
    p8.add_argument("--max-daily-notional", type=float, default=5000.0)
    p8.add_argument("--max-orders-per-day", type=int, default=20)
    p8.add_argument("--max-position-notional", type=float, default=2000.0)
    p8.add_argument("--broker-mode", default=BROKER_MODE_MCP, choices=[BROKER_MODE_MCP])
    p8.add_argument("--db", default="sqlite:///quantflow.db")
    p8.set_defaults(func=cmd_propose_order)

    p9 = sub.add_parser("ops-refresh")
    p9.add_argument("--db", default="sqlite:///quantflow.db")
    p9.add_argument("--stale-days", type=int, default=7)
    p9.add_argument("--force", action="store_true")
    p9.add_argument("--with-report", action="store_true")
    p9.add_argument("--include-api", action="store_true")
    p9.add_argument("--api-base", default="http://127.0.0.1:8100")
    p9.add_argument("--report-out", default="docs/LOCAL_SYSTEM_REPORT.md")
    p9.set_defaults(func=cmd_ops_refresh)

    p10 = sub.add_parser("ops-report")
    p10.add_argument("--db", default="sqlite:///quantflow.db")
    p10.add_argument("--stale-days", type=int, default=7)
    p10.add_argument("--include-api", action="store_true")
    p10.add_argument("--api-base", default="http://127.0.0.1:8100")
    p10.add_argument("--report-out", default="docs/LOCAL_SYSTEM_REPORT.md")
    p10.set_defaults(func=cmd_ops_report)

    p11 = sub.add_parser("train-features")
    p11.add_argument("--tickers", default="")
    p11.add_argument("--db", default="sqlite:///quantflow.db")
    p11.add_argument("--period", default="2y")
    p11.add_argument("--interval", default="1d")
    p11.add_argument("--lookback-days", type=int, default=30)
    p11.add_argument("--forecast-horizon", type=int, default=20)
    p11.add_argument("--save", action="store_true")
    p11.add_argument("--out", default="")
    p11.set_defaults(func=cmd_train_features)

    p12 = sub.add_parser("news-ingest")
    p12.add_argument("--input", required=True)
    p12.add_argument("--db", default="sqlite:///quantflow.db")
    p12.add_argument("--out", default="")
    p12.set_defaults(func=cmd_news_ingest)

    p13 = sub.add_parser("export-table")
    p13.add_argument("--table", required=True)
    p13.add_argument("--db", default="sqlite:///quantflow.db")
    p13.add_argument("--out", required=True)
    p13.set_defaults(func=cmd_export_table)

    p14 = sub.add_parser("build-dataset")
    p14.add_argument("--tickers", default="")
    p14.add_argument("--sector", default="", help="Sector key from SECTOR_UNIVERSE (e.g., 'Financials')")
    p14.add_argument("--period", default="5y")
    p14.add_argument("--interval", default="1d")
    p14.add_argument("--lookback", type=int, default=60)
    p14.add_argument("--forecast-horizon", type=int, default=21)
    p14.add_argument("--snapshot-step", type=int, default=3)
    p14.add_argument("--max-rows", type=int, default=0)
    p14.add_argument("--out", default="")
    p14.set_defaults(func=cmd_build_dataset)

    p15 = sub.add_parser("train")
    p15.add_argument("--tickers", default="")
    p15.add_argument("--period", default="5y")
    p15.add_argument("--interval", default="1d")
    p15.add_argument("--snapshot-step", type=int, default=3)
    p15.add_argument("--lookback", type=int, default=60)
    p15.add_argument("--forecast-horizon", type=int, default=21)
    p15.add_argument("--max-rows", type=int, default=0)
    p15.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    p15.add_argument("--hidden-dim", type=int, default=128)
    p15.add_argument("--dropout", type=float, default=0.2)
    p15.add_argument("--epochs", type=int, default=50)
    p15.add_argument("--batch-size", type=int, default=64)
    p15.add_argument("--learning-rate", type=float, default=1e-4)
    p15.add_argument("--no-multi-scale", action="store_true")
    p15.add_argument("--no-coherent-heads", action="store_true")
    p15.add_argument("--cls-weight", type=float, default=0.35)
    p15.add_argument("--reg-weight", type=float, default=0.25)
    p15.add_argument("--coh-weight", type=float, default=0.10)
    p15.add_argument("--checkpoint-dir", default="checkpoints")
    p15.add_argument("--device", default="")
    p15.add_argument("--save", default="")
    p15.set_defaults(func=cmd_train)

    p16 = sub.add_parser("ai-backtest")
    p16.add_argument("--tickers", default="")
    p16.add_argument("--model", required=True, help="Path to saved model .pt file")
    p16.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    p16.add_argument("--hidden-dim", type=int, default=128)
    p16.add_argument("--lookback", type=int, default=60)
    p16.add_argument("--forecast-horizon", type=int, default=21)
    p16.add_argument("--no-multi-scale", action="store_true")
    p16.add_argument("--no-coherent-heads", action="store_true")
    p16.add_argument("--period", default="5y")
    p16.add_argument("--interval", default="1d")
    p16.add_argument("--start", default="2020-01-01")
    p16.add_argument("--entry-threshold", type=float, default=0.60)
    p16.add_argument("--exit-threshold", type=float, default=0.40)
    p16.add_argument("--max-hold-days", type=int, default=21)
    p16.add_argument("--stop-loss-pct", type=float, default=0.05)
    p16.add_argument("--take-profit-pct", type=float, default=0.0)
    p16.add_argument("--device", default="")
    p16.add_argument("--viz", action="store_true", help="Generate visualization charts")
    p16.add_argument("--viz-dir", default="reports", help="Output directory for visualization charts")
    p16.set_defaults(func=cmd_ai_backtest)

    p17 = sub.add_parser("predict")
    p17.add_argument("--tickers", default="")
    p17.add_argument("--model", required=True, help="Path to saved model .pt file")
    p17.add_argument("--arch", default="bilstm", choices=["bilstm", "transformer", "tcn"])
    p17.add_argument("--hidden-dim", type=int, default=128)
    p17.add_argument("--lookback", type=int, default=60)
    p17.add_argument("--forecast-horizon", type=int, default=21)
    p17.add_argument("--no-multi-scale", action="store_true")
    p17.add_argument("--no-coherent-heads", action="store_true")
    p17.add_argument("--period", default="2y")
    p17.add_argument("--device", default="")
    p17.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
