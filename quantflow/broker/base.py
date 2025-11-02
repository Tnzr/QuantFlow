from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from .types import Position, Account


class Broker(ABC):
    @abstractmethod
    def login(self) -> None:
        ...

    @abstractmethod
    def account(self) -> Account:
        ...

    @abstractmethod
    def positions(self) -> List[Position]:
        ...
