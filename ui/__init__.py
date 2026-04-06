"""
Textual TUI module placeholder.

This module is reserved for future Textual-based terminal UI implementation.
Currently low priority. Interface is defined but not implemented.
"""

from abc import ABC, abstractmethod
from typing import Any


class ITUI(ABC):
    """TUI abstract interface."""

    @abstractmethod
    async def start(self) -> None:
        """Start TUI."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop TUI."""

    @abstractmethod
    def update_status(self, data: dict[str, Any]) -> None:
        """Update status display."""
