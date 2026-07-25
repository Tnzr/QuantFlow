from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_dev_env(dotenv_path: Optional[str] = None) -> bool:
    """Load local .env into process environment when available.

    This is intentionally best-effort for developer ergonomics:
    - no error if python-dotenv is not installed
    - no error if .env does not exist
    - existing environment variables are not overridden
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return False

    if dotenv_path:
        target = Path(dotenv_path)
    else:
        target = Path(__file__).resolve().parents[1] / ".env"

    if not target.exists():
        return False

    load_dotenv(dotenv_path=target, override=False)
    return True
