"""Backward-compatible import path for Technology One.

New code should import from ``technologies.technology_one``. Existing server
profiles and commands can continue to use ``local_experiments.lego``.
"""

from pathlib import Path

__path__ = [
    str(Path(__file__).resolve().parents[2] / "technologies" / "technology_one")
]
