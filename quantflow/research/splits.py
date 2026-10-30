"""Purged and embargoed cross-validation for overlapping financial labels.

A sample observed at ``t`` with horizon ``h`` uses information up to
``t + h`` bars. Any training sample whose label window overlaps the test
window is purged, and samples within ``embargo`` bars after the test window
are dropped so serially correlated features cannot leak test information.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _dates_of(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(index.get_level_values(0))
    return pd.DatetimeIndex(index)


def _bar_positions(calendar: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> np.ndarray:
    return calendar.searchsorted(dates, side="left")


def purge_train(
    sample_pos: np.ndarray,
    train_mask: np.ndarray,
    test_blocks: list[tuple[int, int]],
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return a train mask with label-overlap and embargo samples removed.

    ``test_blocks`` are inclusive ``(first_bar, last_bar)`` positions on the
    trading calendar.
    """
    keep = train_mask.copy()
    label_end = sample_pos + horizon
    for lo, hi in test_blocks:
        overlaps = (label_end >= lo) & (sample_pos <= hi)
        embargoed = (sample_pos > hi) & (sample_pos <= hi + embargo)
        keep &= ~(overlaps | embargoed)
    return keep


class PurgedWalkForward:
    """Expanding (or rolling) walk-forward split, one test block per period.

    Args:
        horizon: label horizon in trading bars.
        embargo: extra bars dropped after each test block (defaults to horizon).
        freq: pandas period alias for test blocks ("Y", "Q", "M").
        min_train_bars: skip folds with less training history than this.
        max_train_bars: if set, use a rolling window of this many bars.
    """

    def __init__(
        self,
        horizon: int,
        embargo: int | None = None,
        freq: str = "Y",
        min_train_bars: int = 756,
        max_train_bars: int | None = None,
    ):
        self.horizon = int(horizon)
        self.embargo = self.horizon if embargo is None else int(embargo)
        self.freq = freq
        self.min_train_bars = int(min_train_bars)
        self.max_train_bars = max_train_bars

    def split(self, index: pd.Index, calendar: pd.DatetimeIndex | None = None) -> Iterator[Fold]:
        dates = _dates_of(index)
        cal = pd.DatetimeIndex(calendar if calendar is not None else dates.unique()).sort_values()
        pos = _bar_positions(cal, dates)
        periods = dates.to_period(self.freq)
        for per in periods.unique().sort_values():
            test_mask = np.asarray(periods == per)
            lo, hi = int(pos[test_mask].min()), int(pos[test_mask].max())
            if lo < self.min_train_bars:
                continue
            train_mask = pos < lo
            if self.max_train_bars:
                train_mask &= pos >= lo - int(self.max_train_bars)
            train_mask = purge_train(pos, train_mask, [(lo, hi)], self.horizon, self.embargo)
            if not train_mask.any():
                continue
            yield Fold(np.flatnonzero(train_mask), np.flatnonzero(test_mask), cal[lo], cal[hi])


class CombinatorialPurgedKFold:
    """Combinatorial purged CV (Lopez de Prado): C(n_groups, n_test) splits.

    Training samples on both sides of each test group are allowed, after
    purging overlaps and embargoing the bars that follow each test group.
    Used to build a distribution of OOS Sharpe ratios rather than one path.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, horizon: int = 5, embargo: int | None = None):
        if not 0 < n_test_groups < n_groups:
            raise ValueError("need 0 < n_test_groups < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.horizon = int(horizon)
        self.embargo = self.horizon if embargo is None else int(embargo)

    def split(self, index: pd.Index, calendar: pd.DatetimeIndex | None = None) -> Iterator[Fold]:
        dates = _dates_of(index)
        cal = pd.DatetimeIndex(calendar if calendar is not None else dates.unique()).sort_values()
        pos = _bar_positions(cal, dates)
        edges = np.linspace(pos.min(), pos.max() + 1, self.n_groups + 1).astype(int)
        groups = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(self.n_groups)]
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            blocks = [groups[i] for i in combo]
            test_mask = np.zeros(len(pos), dtype=bool)
            for lo, hi in blocks:
                test_mask |= (pos >= lo) & (pos <= hi)
            train_mask = purge_train(pos, ~test_mask, blocks, self.horizon, self.embargo)
            if not train_mask.any() or not test_mask.any():
                continue
            yield Fold(
                np.flatnonzero(train_mask),
                np.flatnonzero(test_mask),
                cal[min(b[0] for b in blocks)],
                cal[min(max(b[1] for b in blocks), len(cal) - 1)],
            )
