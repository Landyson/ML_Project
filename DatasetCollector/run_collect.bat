@echo off
REM Spuštění sběru dat (Windows) - Binance varianta (2000+ záznamů)
REM 1) python -m venv .venv
REM 2) .venv\Scripts\activate
REM 3) pip install -r requirements.txt
REM 4) spusť:

python src\collect_binance_klines.py --pair XBTUSD --interval 1h --rows 2200

echo.
echo Vystup:
echo - out\data.csv
pause
