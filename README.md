# Code for "Resilience, Load Forecasting, and Renewable Penetration in Isolated Power Systems: Evidence from the NOGA-ISO Grid"

Release tag: `scirep-revision-1`. Python 3.14.5 (results identical with 3.12); versions in `requirements.txt`.

Place the scripts in the same folder as the data files (`noga_demand_5min.csv`, `noga_production_mix_5min.csv`, `noga_co2_5min.csv`, `weather_tel_aviv_hourly.csv`). Every script reads its inputs from its own folder and writes its outputs there.

| Step | Scripts |
|---|---|
| Data collection (Playwright, real Chromium, reCAPTCHA) | `noga_co2_demand_download.py`, `noga_piecharts_download.py` |
| Filtering, forecast-error statistics, security events | `forecast_error_analysis.py`, `compute_statistics.py`, `full_grid_analysis.py` |
| Persistence baseline | `compute_persistence.py` |
| Short-term model LGB-SF-W (weather download and alignment, features, training, testing) | `forecast_ml_v2.py` (earlier versions: `forecast_ml.py`, `forecast_dl.py`) |
| Statistical tests and model checks added in the revision | `revision_statistics.py`, `revision_model_checks.py` |
| Figures | `fig_*.py`, `generate_penetration_plot.py` |
