from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Sector-organized investment universe for QuantFlow-AI-Core training
# Diverse sectors with liquid, optionable tickers.
# ---------------------------------------------------------------------------

SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "Technology / AI": ["NVDA", "MSFT", "AAPL", "AVGO"],
    "Technology / AI — Alternatives": ["AMD", "ORCL", "META", "GOOGL"],
    "Semiconductors": ["NVDA", "AMD", "TSM", "ASML"],
    "Semiconductors — Alternatives": ["MU", "QCOM", "ARM", "AMAT", "SMCI", "LRCX", "MRVL", "INTC"],
    "Cloud / Software": ["MSFT", "AMZN", "ORCL", "CRM"],
    "Cloud / Software — Alternatives": ["SNOW", "DDOG", "PANW", "NOW", "ADBE", "MDB"],
    "Financials": ["JPM", "GS", "BAC", "V"],
    "Financials — Alternatives": ["MA", "BRK.B", "SCHW", "AXP", "BLK", "MS"],
    "Consumer": ["AMZN", "COST", "WMT", "HD"],
    "Consumer — Alternatives": ["TGT", "LOW", "TJX", "SBUX", "NKE", "MCD"],
    "Communication": ["META", "GOOGL", "NFLX", "DIS"],
    "Communication — Alternatives": ["T", "VZ", "CMCSA", "CHTR"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Energy — Alternatives": ["EOG", "PSX", "MPC", "PXD", "OXY"],
    "Industrials": ["CAT", "GE", "DE", "HON"],
    "Industrials — Alternatives": ["UNP", "UPS", "LMT", "RTX", "BA", "ETN"],
    "Healthcare": ["LLY", "UNH", "JNJ", "ABBV"],
    "Healthcare — Alternatives": ["MRK", "ISRG", "PFE", "TMO", "DHR", "ABT"],
    "Biotechnology": ["AMGN", "REGN", "GILD", "VRTX"],
    "Biotechnology — Alternatives": ["BIIB", "MRNA", "ILMN", "ALNY"],
    "Aerospace & Defense": ["RTX", "LMT", "NOC", "GE"],
    "Aerospace & Defense — Alternatives": ["BA", "GD", "LHX", "HWM"],
    "Real Estate": ["PLD", "AMT", "EQIX", "O"],
    "Real Estate — Alternatives": ["SPG", "CCI", "WELL", "DLR"],
    "Utilities": ["NEE", "DUK", "SO", "AEP"],
    "Utilities — Alternatives": ["D", "EXC", "SRE", "XEL"],
    "Transportation": ["UPS", "FDX", "UNP", "CSX"],
    "Transportation — Alternatives": ["NSC", "DAL", "LUV", "UBER"],
    "Automotive / EV": ["TSLA", "GM", "F", "RIVN"],
    "Automotive / EV — Alternatives": ["LCID", "HMC", "TM", "STLA"],
    "Internet": ["GOOGL", "META", "UBER", "SPOT"],
    "China ADRs": ["BABA", "JD", "PDD", "BIDU"],
    "China ADRs — Alternatives": ["NIO", "TCOM", "NTES", "BILI"],
    # -----------------------------------------------------------------------
    # Broad-market ETFs for macro context & hedging signals
    # -----------------------------------------------------------------------
    "Broad ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "XLE", "XLF", "XLK", "XLV", "SMH"],
}

# Flat deduplicated list for convenience (scan / recommend / train-features)
HIGH_INTEREST: List[str] = sorted(set(
    ticker
    for names in SECTOR_UNIVERSE.values()
    for ticker in names
))