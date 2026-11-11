"""
Download NOGA-ISO CO2 emissions and demand-curve data for all available days.
Produces two CSV files (5-minute intervals):
  ~/noga_co2_5min.csv
  ~/noga_demand_5min.csv

Date ranges:
  CO2:    2023-06-01 → 2025-12-31
  Demand: 2023-04-01 → 2025-12-31
"""

import asyncio, csv, json, sys
from datetime import date, timedelta
from pathlib import Path

PAGES = {
    "co2": {
        "url":       "https://www.noga-iso.co.il/systemoperationunit/co2/",
        "api_key":   "Co2Graph/Get",
        "js_var":    "formattedCo2DateForService",
        "js_fn":     "CreateCo2Grpahs()",
        "start":     date(2023, 6, 1),
        "end":       date(2025, 12, 31),
        "col_map": {
            "date":          "Date",
            "time":          "Time",
            "order":         "Interval (1-288)",
            "allSites":      "Total Emissions (mTCO2/h)",
            "coal":          "Coal Emissions (mTCO2/h)",
            "gas":           "Gas Emissions (mTCO2/h)",
            "diesel":        "Diesel Emissions (mTCO2/h)",
            "mazut":         "Mazut Emissions (mTCO2/h)",
            "methanol":      "Methanol Emissions (mTCO2/h)",
            "renewables":    "Renewables Emissions (mTCO2/h)",
            "renewSave":     "Renewable Savings (mTCO2/h)",
            "ratio":         "Emission Ratio (tCO2/MWh)",
            "currentDemand": "Current Demand (MW)",
        },
        "out_csv": Path.home() / "noga_co2_5min.csv",
    },
    "demand": {
        "url":       "https://www.noga-iso.co.il/systemoperationunit/demand-curve/",
        "api_key":   "DemandGraph/Get",
        "js_var":    "formattedEconomicDemandForecastDateForService",
        "js_fn":     "CreateEconomicDemandForecastGrpahs()",
        "js_clear":  "$('#economicDemandForecastGraphDate').val('');",  # clear picker so var takes over
        "start":     date(2023, 4, 1),
        "end":       date(2025, 12, 31),
        "col_map": {
            "date":           "Date",
            "time":           "Time",
            "order":          "Interval (1-288)",
            "currentDemand":  "Current Demand (MW)",
            "updatedDemand":  "Updated Demand Forecast (MW)",
            "dayAheadDemand": "Day-Ahead Demand Forecast (MW)",
        },
        "out_csv": Path.home() / "noga_demand_5min.csv",
    },
}


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


async def download_page(cfg: dict):
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    api_key   = cfg["api_key"]
    js_var    = cfg["js_var"]
    js_fn     = cfg["js_fn"]
    js_clear  = cfg.get("js_clear", "")
    col_map   = cfg["col_map"]
    out_csv   = cfg["out_csv"]

    all_dates = [d for d in date_range(cfg["start"], cfg["end"]) if d.weekday() != 5]

    # Resume: find dates already in CSV
    done_dates = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done_dates.add(row.get("Date", ""))

    pending = [d for d in all_dates if d.isoformat() not in done_dates]
    print(f"\n{'='*60}")
    print(f"Dataset: {api_key}")
    print(f"Total working days: {len(all_dates)}  |  Already done: {len(done_dates)}  |  Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Open CSV in append mode (write header only if new file)
    is_new = not out_csv.exists() or out_csv.stat().st_size == 0
    fout = open(out_csv, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fout, fieldnames=list(col_map.values()))
    if is_new:
        writer.writeheader()

    api_store = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1400, "height": 900}, locale="he-IL")

        async def capture(response):
            if api_key in response.url:
                try:
                    api_store["latest"] = await response.json()
                except Exception:
                    pass
        page.on("response", capture)

        print(f"Loading page…")
        await page.goto(cfg["url"], wait_until="load", timeout=60000)
        await page.wait_for_timeout(6000)   # wait for initial chart

        # Remove cookie banner
        await page.evaluate("""
            () => document.querySelectorAll(
                '[id*="silktide"],[class*="silktide"],[id*="cookie"],[class*="cookie-banner"],[class*="consent-banner"],[id*="consent"]'
            ).forEach(el => el.remove())
        """)

        errors = []
        for i, d in enumerate(pending):
            date_iso = d.isoformat()
            api_store.pop("latest", None)

            for attempt in range(3):  # up to 3 attempts per date
                try:
                    api_store.pop("latest", None)
                    await page.evaluate(f"""
                        () => new Promise((resolve, reject) => {{
                            {js_clear}
                            window['{js_var}'] = '{date_iso}';
                            {js_fn.rstrip(';')}.then(resolve).catch(reject);
                        }})
                    """)
                    await page.wait_for_timeout(1000)

                    data = api_store.get("latest")
                    if data and isinstance(data, list) and len(data) > 0:
                        for rec in data:
                            row = {col_name: (rec.get(src_key) or 0)
                                   for src_key, col_name in col_map.items()}
                            writer.writerow(row)
                        fout.flush()
                        print(f"  [{i+1}/{len(pending)}] {date_iso}  ({len(data)} records)")
                        break  # success
                    else:
                        raise ValueError("empty response")

                except (PWTimeout, Exception) as exc:
                    if attempt < 2:
                        wait_ms = 5000 * (attempt + 1)  # 5s then 10s
                        print(f"  [{i+1}/{len(pending)}] {date_iso}  attempt {attempt+1} failed ({exc}) — retrying in {wait_ms//1000}s")
                        await page.wait_for_timeout(wait_ms)
                    else:
                        print(f"  [{i+1}/{len(pending)}] {date_iso}  FAILED after 3 attempts")
                        errors.append(date_iso)

            await page.wait_for_timeout(800)

        await browser.close()

    fout.close()
    print(f"\nWrote: {out_csv}")
    if errors:
        print(f"Failed ({len(errors)}): {errors[:10]}{'...' if len(errors)>10 else ''}")


async def main():
    for key in ["co2", "demand"]:
        await download_page(PAGES[key])
    print("\nAll done.")


if __name__ == "__main__":
    asyncio.run(main())
