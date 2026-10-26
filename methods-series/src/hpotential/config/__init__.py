"""Frozen configurations holding the exact published parameters.

Each analysis takes a configuration and defaults to the published one, so a
reader can vary a design without editing the code and a reviewer can see every
constant in one place.
"""

from __future__ import annotations

from .paper1 import Paper1Config
from .paper2 import Paper2Config
from .paper3 import Paper3Config
from .paper4 import Paper4Config
from .paper5 import Paper5Config
from .paper6 import Paper6Config

__all__ = [
    "Paper1Config",
    "Paper2Config",
    "Paper3Config",
    "Paper4Config",
    "Paper5Config",
    "Paper6Config",
]
