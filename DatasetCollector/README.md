# Data Collector — XBTUSD (Binance klines)

Tahle část projektu slouží ke **sběru reálných tržních dat** (OHLCV svíčky) a vytvoření datasetu pro ML.

## Proč Binance (a ne Kraken OHLC)
Kraken endpoint pro OHLC vrací pouze ~720 nejnovějších svíček. Pro školní požadavek 1500+ záznamů používáme Binance klines, které umí získat delší historii.

## Co skript vytvoří
- `out/data.csv` (finální dataset)
- `raw/` (raw JSON odpovědi z API jako důkaz původu dat)

Sloupce v datasetu:
- `timestamp_ms`, `datetime_utc`, `instrument`
- `open`, `high`, `low`, `close`, `volume`
- `rsi_14`
- `direction` (RSI <= 50 -> long, jinak short)
- `profit` (0/1 podle pohybu close o 1 svíčku dopředu)

## Spuštění (bez IDE)
### 1) Vytvoření prostředí
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Sběr dat (hodinová data)
```bat
python src\collect_binance_klines.py --pair XBTUSD --interval 1h --rows 2200
```

Výstup najdeš v `out/data.csv`.

> Pozn.: `--pair XBTUSD` se mapuje na Binance symbol `BTCUSDT` (nejběžnější spot trh). V datasetu necháváme `instrument` jako XBTUSD.
