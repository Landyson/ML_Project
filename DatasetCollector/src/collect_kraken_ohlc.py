"""
collect_kraken_ohlc.py
Autorský skript pro sběr reálných dat z Kraken API (OHLC).

Výstupy:
1) out/data.csv
   - "bohatý" dataset s OHLCV + RSI + forward return + label

2) out/simple_trades.csv  (BEZ HLAVIČKY)
   - jednoduchý formát blízký "trade logu":
     instrument,side,rsi,volume_category,day_part,label

   Příklad řádku:
   XBTUSD,long,62,high,evening,1

Poznámka k obhajobě:
- "side" a "label" nejsou skutečné obchody uživatele.
  Jsou to signály + vyhodnocení, zda by signál vyšel podle reálného pohybu ceny.
  To je legitimní ML úloha: predikce úspěšnosti signálu.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests


KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"


@dataclass
class CollectorConfig:
    pair: str
    interval: int
    rows: int
    output_csv: Path
    output_simple_csv: Path
    raw_dir: Path
    fwd_minutes: int


def parse_args() -> CollectorConfig:
    p = argparse.ArgumentParser(description="Sběr OHLC dat z Kraken API a export do CSV.")
    p.add_argument("--pair", type=str, default="XBTUSD", help="Kraken pair, např. XBTUSD, ETHUSD")
    p.add_argument("--interval", type=int, default=1, help="Interval v minutách (1, 5, 15, 60, ...)")
    p.add_argument("--rows", type=int, default=2000, help="Kolik záznamů (svíček) chceme aspoň nasbírat")
    p.add_argument("--fwd", type=int, default=5, help="Kolik minut dopředu hodnotíme výsledek signálu (default 5)")
    p.add_argument("--output", type=str, default="out/data.csv", help="Výstupní CSV (bohatý dataset)")
    p.add_argument("--output-simple", type=str, default="out/simple_trades.csv", help="Výstupní jednoduché CSV (bez hlavičky)")
    p.add_argument("--raw-dir", type=str, default="raw", help="Složka pro ukládání původních JSON odpovědí")
    args = p.parse_args()
    return CollectorConfig(
        pair=args.pair,
        interval=args.interval,
        rows=args.rows,
        output_csv=Path(args.output),
        output_simple_csv=Path(args.output_simple),
        raw_dir=Path(args.raw_dir),
        fwd_minutes=args.fwd,
    )


def kraken_fetch_ohlc(pair: str, interval: int, since: int | None = None) -> Dict[str, Any]:
    params = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since
    resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=30)
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


def day_part(hour_utc: int) -> str:
    if 6 <= hour_utc <= 11:
        return "morning"
    if 12 <= hour_utc <= 17:
        return "afternoon"
    if 18 <= hour_utc <= 22:
        return "evening"
    return "night"


def volume_category(series_volume: pd.Series) -> pd.Series:
    """
    Rozdělí volume na low/medium/high podle kvantilů (33% / 66%).
    """
    q1 = series_volume.quantile(0.33)
    q2 = series_volume.quantile(0.66)

    def cat(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    return series_volume.apply(cat)


def build_dataset(all_rows: List[List[Any]], instrument: str, interval_minutes: int, fwd_minutes: int) -> pd.DataFrame:
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = df["timestamp"].astype(int)
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)

    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).astype(str)
    df["instrument"] = instrument
    df["hour_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour

    df["rsi_14"] = compute_rsi(df["close"], period=14)

    # Převod "minuty dopředu" na počet řádků podle intervalu
    # interval=1 -> fwd 5 => 5 řádků; interval=5 -> fwd 5 => 1 řádek
    step = max(1, int(round(fwd_minutes / interval_minutes)))
    df["return_fwd"] = (df["close"].shift(-step) - df["close"]) / df["close"]

    # Bohatý label: pouze růst ceny
    df["label_up_fwd"] = (df["return_fwd"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def build_simple_format(df: pd.DataFrame, fwd_minutes: int) -> pd.DataFrame:
    """
    Vytvoří jednoduchý formát:
    instrument,side,rsi,volume_category,day_part,label

    side (signál) pravidlo:
      - RSI < 45 => long
      - RSI > 55 => short
      - jinak žádný signál => řádek vynecháme

    label:
      - pro long: 1 pokud return_fwd > 0
      - pro short: 1 pokud return_fwd < 0
    """
    out = df.copy()

    # Zaokrouhlené RSI (čitelnost)
    out["rsi"] = out["rsi_14"].round().astype(int)

    out["volume_cat"] = volume_category(out["volume"])
    out["day_part"] = out["hour_utc"].apply(day_part)

    # side podle RSI
    conditions = [
        out["rsi_14"] < 45,
        out["rsi_14"] > 55,
    ]
    choices = ["long", "short"]
    out["side"] = np.select(conditions, choices, default="")

    # necháme jen řádky, kde máme signál
    out = out[out["side"] != ""].copy()

    # label podle směru
    out["label"] = 0
    out.loc[(out["side"] == "long") & (out["return_fwd"] > 0), "label"] = 1
    out.loc[(out["side"] == "short") & (out["return_fwd"] < 0), "label"] = 1

    # finální sloupce v pořadí
    simple = out[["instrument", "side", "rsi", "volume_cat", "day_part", "label"]].reset_index(drop=True)
    return simple


def main() -> None:
    cfg = parse_args()
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_simple_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[List[Any]] = []
    since = None
    safety = 0

    while len(all_rows) < cfg.rows:
        safety += 1
        if safety > 50:
            break

        payload = kraken_fetch_ohlc(cfg.pair, cfg.interval, since=since)
        if payload.get("error"):
            raise RuntimeError(f"Kraken API error: {payload['error']}")

        result = payload.get("result", {})
        last = result.get("last")
        pair_key = next((k for k in result.keys() if k != "last"), None)
        if pair_key is None:
            raise RuntimeError("Neočekávaný formát odpovědi Kraken API.")

        rows = result[pair_key]
        if not rows:
            break

        # Ulož raw odpověď (důkaz původu dat)
        ts = int(time.time())
        raw_path = cfg.raw_dir / f"kraken_ohlc_{cfg.pair}_{cfg.interval}m_{ts}.json"
        raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        all_rows.extend(rows)
        since = last
        time.sleep(0.3)

        if len(rows) < 10:
            break

    # deduplikace a seřazení
    tmp = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    tmp["timestamp"] = tmp["timestamp"].astype(int)
    tmp = tmp.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = build_dataset(tmp.values.tolist(), instrument=cfg.pair, interval_minutes=cfg.interval, fwd_minutes=cfg.fwd_minutes)
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8")

    simple = build_simple_format(df, fwd_minutes=cfg.fwd_minutes)

    # Jednoduchý formát uložíme BEZ HLAVIČKY přesně jak chceš:
    # XBTUSD,long,62,high,evening,1
    simple.to_csv(cfg.output_simple_csv, index=False, header=False, encoding="utf-8")

    print("Hotovo.")
    print(f"- data.csv:         {cfg.output_csv} | řádků: {len(df)}")
    print(f"- simple_trades.csv:{cfg.output_simple_csv} | řádků: {len(simple)}")


if __name__ == "__main__":
    main()
