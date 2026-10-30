"""US federal election calendar, outcomes and presidential-cycle helpers.

Outcomes are public historical facts; they are only ever used in backtests
for bars dated *after* the corresponding election day (see ``known_outcomes_as_of``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Election:
    day: date
    kind: str  # "midterm" | "presidential"
    president_party: str  # party holding the White House on election day
    senate_before: str
    senate_after: Optional[str]
    house_before: str
    house_after: Optional[str]

    @property
    def pres_party_lost_senate(self) -> Optional[bool]:
        if self.senate_after is None:
            return None
        return self.senate_before == self.president_party and self.senate_after != self.president_party

    @property
    def divided_after(self) -> Optional[bool]:
        if self.senate_after is None or self.house_after is None:
            return None
        return not (self.senate_after == self.house_after == self.president_party)

    @property
    def label(self) -> str:
        return f"{self.day.year} {self.kind}"


ELECTIONS: list[Election] = [
    Election(date(2000, 11, 7), "presidential", "D", "R", "R", "R", "R"),
    Election(date(2002, 11, 5), "midterm", "R", "D", "R", "R", "R"),
    Election(date(2004, 11, 2), "presidential", "R", "R", "R", "R", "R"),
    Election(date(2006, 11, 7), "midterm", "R", "R", "D", "R", "D"),
    Election(date(2008, 11, 4), "presidential", "R", "D", "D", "D", "D"),
    Election(date(2010, 11, 2), "midterm", "D", "D", "D", "D", "R"),
    Election(date(2012, 11, 6), "presidential", "D", "D", "D", "R", "R"),
    Election(date(2014, 11, 4), "midterm", "D", "D", "R", "R", "R"),
    Election(date(2016, 11, 8), "presidential", "D", "R", "R", "R", "R"),
    Election(date(2018, 11, 6), "midterm", "R", "R", "R", "R", "D"),
    Election(date(2020, 11, 3), "presidential", "R", "R", "D", "D", "D"),
    Election(date(2022, 11, 8), "midterm", "D", "D", "D", "D", "R"),
    Election(date(2024, 11, 5), "presidential", "D", "D", "R", "R", "R"),
    Election(date(2026, 11, 3), "midterm", "R", "R", None, "R", None),
]

CYCLE_PHASES = {0: "election", 1: "post_election", 2: "midterm", 3: "pre_election"}


def cycle_phase(year: int) -> str:
    """Presidential-cycle phase: election (yr%4==0), post_election, midterm, pre_election."""
    return CYCLE_PHASES[year % 4]


def is_election_year(year: int) -> bool:
    return year % 4 == 0


def elections(kind: Optional[str] = None, resolved_only: bool = False) -> list[Election]:
    out = [e for e in ELECTIONS if kind is None or e.kind == kind]
    if resolved_only:
        out = [e for e in out if e.senate_after is not None]
    return out


def next_election(today: Optional[date] = None) -> Optional[Election]:
    today = today or date.today()
    upcoming = [e for e in ELECTIONS if e.day >= today]
    return upcoming[0] if upcoming else None


def days_to_next_election(as_of) -> Optional[int]:
    d = pd.Timestamp(as_of).date()
    e = next_election(d)
    return None if e is None else (e.day - d).days


def election_in_window(as_of, dte: int) -> bool:
    days = days_to_next_election(as_of)
    return days is not None and 0 <= days <= dte


def known_outcomes_as_of(as_of) -> list[Election]:
    """Elections whose outcome was public at ``as_of`` (day after election)."""
    d = pd.Timestamp(as_of).date()
    return [e for e in ELECTIONS if e.senate_after is not None and e.day < d]
