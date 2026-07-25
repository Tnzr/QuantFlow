from __future__ import annotations

import os
import sys

import requests


def main() -> int:
    base = os.getenv("API_BASE", "http://127.0.0.1:8100").rstrip("/")
    checks = [
        ("health", f"{base}/health"),
        ("scraping", f"{base}/scanner/scraping/status"),
        ("report", f"{base}/ops/latest-report"),
        ("env_validate", f"{base}/ops/env/validate"),
    ]

    for name, url in checks:
        try:
            resp = requests.get(url, timeout=20)
        except Exception as exc:
            print(f"SMOKE_FAIL {name} request_error={exc} url={url}")
            return 1

        if resp.status_code != 200:
            print(f"SMOKE_FAIL {name} status={resp.status_code} url={url}")
            print(resp.text[:500])
            return 1

        try:
            payload = resp.json()
        except Exception:
            print(f"SMOKE_FAIL {name} invalid_json url={url}")
            print(resp.text[:500])
            return 1

        print(f"SMOKE_OK {name} keys={sorted(list(payload.keys()))[:6]}")

    print("SMOKE_OK all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
