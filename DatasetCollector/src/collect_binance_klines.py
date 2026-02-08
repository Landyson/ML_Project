from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


BINANCE_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

PAIR_MAP = {
    "XBTUSD": "BTCUSDT",   # mapování na nejběžnější spot symbol
    "BTCUSDT": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "ETHUSDT": "ETHUSDT",
}


@dataclass
class CollectorConfig:
    pair: str
    interval: str
    rows: int
    output_csv: Path
    raw_dir: Path
    rsi_period: int


def parse_args() -> CollectorConfig:
    p = argparse.ArgumentParser(description="Sběr OHLCV dat z Binance (klines) a export datasetu do CSV.")
    p.add_argument("--pair", type=str, default="XBTUSD", help="Např. XBTUSD (mapuje na BTCUSDT), nebo BTCUSDT")
    p.add_argument("--interval", type=str, default="1h", help="Binance interval: 1m,5m,15m,1h,4h,1d,...")
    p.add_argument("--rows", type=int, default=2200, help="Kolik záznamů chceme ve finále (doporuč 2200)")
    p.add_argument("--output", type=str, default="out/data.csv", help="Výstupní CSV (finální dataset)")
    p.add_argument("--raw-dir", type=str, default="raw", help="Složka pro ukládání raw JSON odpovědí")
    p.add_argument("--rsi-period", type=int, default=14, help="Perioda RSI (default 14)")
    a = p.parse_args()

    return CollectorConfig(
        pair=a.pair,
        interval=a.interval,
        rows=a.rows,
        output_csv=Path(a.output),
        raw_dir=Path(a.raw_dir),
        rsi_period=a.rsi_period,
    )


def map_pair_to_symbol(pair: str) -> str:
    key = pair.strip().upper()
    return PAIR_MAP.get(key, key)


def binance_get_klines(symbol: str, interval: str, limit: int = 1000, start_time_ms: Optional[int] = None) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    url = BINANCE_BASE + KLINES_ENDPOINT
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_dataset(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    df["rsi_14"] = compute_rsi(df["close"], period=rsi_period)
    df["direction"] = np.where(df["rsi_14"] <= 50, "long", "short")

    next_close = df["close"].shift(-1)
    df["profit"] = 0
    df.loc[(df["direction"] == "long") & (next_close > df["close"]), "profit"] = 1
    df.loc[(df["direction"] == "short") & (next_close < df["close"]), "profit"] = 1

    df = df.dropna().copy().reset_index(drop=True)
    df["profit"] = df["profit"].astype(int)
    return df


def main() -> None:
    cfg = parse_args()
    symbol = map_pair_to_symbol(cfg.pair)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    interval_to_ms = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    if cfg.interval not in interval_to_ms:
        raise ValueError(f"Nepodporovaný interval '{cfg.interval}'. Použij např. 1h, 4h, 1d, 15m...")

    step_ms = interval_to_ms[cfg.interval]
    reserve = 250  # kvůli RSI a poslednímu řádku
    need_raw = cfg.rows + reserve

    now_ms = int(time.time() * 1000)
    start_time = now_ms - need_raw * step_ms

    all_klines: List[List[Any]] = []
    safety = 0

    while len(all_klines) < need_raw:
        safety += 1
        if safety > 30:
            break

        batch = binance_get_klines(symbol=symbol, interval=cfg.interval, limit=1000, start_time_ms=start_time)
        if not batch:
            break

        ts = int(time.time())
        (cfg.raw_dir / f"binance_klines_{symbol}_{cfg.interval}_{ts}.json").write_text(
            json.dumps(batch, indent=2), encoding="utf-8"
        )

        all_klines.extend(batch)

        last_open_time = int(batch[-1][0])
        start_time = last_open_time + step_ms

        if len(batch) < 1000:
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.rename(columns={"open_time": "timestamp_ms"}, inplace=True)

    df["timestamp_ms"] = df["timestamp_ms"].astype(int)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).astype(str)
    df["instrument"] = cfg.pair.upper()

    df = build_dataset(df, rsi_period=cfg.rsi_period)

    if len(df) > cfg.rows:
        df = df.tail(cfg.rows).reset_index(drop=True)

    df.to_csv(cfg.output_csv, index=False, encoding="utf-8")

    print("Hotovo.")
    print(f"- instrument: {cfg.pair.upper()} (Binance symbol: {symbol})")
    print(f"- interval:   {cfg.interval}")
    print(f"- rows:       {len(df)} (cíleno: {cfg.rows})")
    print(f"- output:     {cfg.output_csv}")


if __name__ == "__main__":
    main()
