"""Thin shim so `python main.py --backtest ...` works from the repo root."""
from state_space_arb.main import main

if __name__ == "__main__":
    raise SystemExit(main())
