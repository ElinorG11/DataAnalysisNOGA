"""
Download all NOGA-ISO production mix pie charts from April 1 2023 to Dec 31 2025.
Saves PNG screenshots (both pie charts) per day to ~/noga_piecharts/.
Uses Playwright with a real browser to pass reCAPTCHA v3 silently.

Usage:
    python3 noga_piecharts_download.py
    python3 noga_piecharts_download.py --headless   # may fail reCAPTCHA
    python3 noga_piecharts_download.py --from 2024-01-01 --to 2024-12-31

Output files:
    ~/noga_piecharts/YYYY-MM-DD_domestic.png    (ייצור משקי)
    ~/noga_piecharts/YYYY-MM-DD_renewable.png   (אנרגיות מתחדשות)
    ~/noga_piecharts/YYYY-MM-DD_data.json       (raw API response)
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path


OUTPUT_DIR = Path.home() / "noga_piecharts"
PAGE_URL = "https://www.noga-iso.co.il/systemoperationunit/piechartspage/"
START_DATE = date(2023, 4, 1)
END_DATE = date(2025, 12, 31)


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


async def run(headless: bool, from_date: date, to_date: date):
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("ERROR: playwright not installed. Run: pip3 install playwright && python3 -m playwright install chromium")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = list(date_range(from_date, to_date))
    # Skip Saturdays (weekday 5) — the site disables them
    work_dates = [d for d in all_dates if d.weekday() != 5]

    # Find already-downloaded dates so we can resume
    done = {
        f.stem.replace("_domestic", "").replace("_renewable", "").replace("_data", "")
        for f in OUTPUT_DIR.glob("*.*")
    }
    pending = [d for d in work_dates if d.isoformat() not in done
               or not (OUTPUT_DIR / f"{d.isoformat()}_domestic.png").exists()
               or not (OUTPUT_DIR / f"{d.isoformat()}_renewable.png").exists()]

    print(f"Total working days: {len(work_dates)}")
    print(f"Already downloaded: {len(work_dates) - len(pending)}")
    print(f"Remaining: {len(pending)}")

    if not pending:
        print("All charts already downloaded.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="he-IL",
        )
        page = await context.new_page()

        # Intercept API to also save raw JSON
        api_data_store = {}

        async def handle_response(response):
            if "ProductionMixGraph/Get" in response.url:
                try:
                    body = await response.json()
                    api_data_store["latest"] = body
                except Exception:
                    pass

        page.on("response", handle_response)

        print(f"\nLoading page (this may take ~15s)...")
        await page.goto(PAGE_URL, wait_until="load", timeout=60000)

        # Wait for the initial charts to finish rendering
        try:
            await page.wait_for_selector("#farm-production-graph .highcharts-root", timeout=30000)
            print("Initial charts loaded OK.\n")
        except PWTimeout:
            print("WARNING: Initial chart load timed out — reCAPTCHA may have blocked.")
            print("Try running without --headless (open browser visible).")
            await browser.close()
            return

        # Dismiss cookie consent banner — remove it via JS so it doesn't overlap screenshots
        try:
            await page.evaluate("""
                () => {
                    // Remove any silktide or cookie banner elements
                    document.querySelectorAll(
                        '[id*="silktide"], [class*="silktide"], [id*="cookie"], [class*="cookie-banner"], ' +
                        '[class*="consent-banner"], [id*="consent"]'
                    ).forEach(el => el.remove());
                }
            """)
            await page.wait_for_timeout(300)
        except Exception:
            pass

        errors = []
        for i, d in enumerate(pending):
            date_iso = d.isoformat()
            domestic_path = OUTPUT_DIR / f"{date_iso}_domestic.png"
            renewable_path = OUTPUT_DIR / f"{date_iso}_renewable.png"
            data_path = OUTPUT_DIR / f"{date_iso}_data.json"

            if domestic_path.exists() and renewable_path.exists():
                continue

            api_data_store.pop("latest", None)

            try:
                # Set the date in JavaScript and trigger the API call
                await page.evaluate(f"""
                    () => new Promise((resolve, reject) => {{
                        window.formattedProductionMixDateForService = '{date_iso}';
                        CreateProdMixGrpahs()
                            .then(resolve)
                            .catch(reject);
                    }})
                """)

                # Give Highcharts time to finish rendering
                await page.wait_for_timeout(1500)

                # Screenshot each pie chart element
                farm_el = page.locator("#farm-production-graph")
                renewable_el = page.locator("#renewable-energies-graph")

                await farm_el.screenshot(path=str(domestic_path))
                await renewable_el.screenshot(path=str(renewable_path))

                # Save raw JSON
                if "latest" in api_data_store:
                    data_path.write_text(json.dumps(api_data_store["latest"], ensure_ascii=False, indent=2))

                status = f"[{i+1}/{len(pending)}] {date_iso} OK"
                print(status)

            except PWTimeout:
                msg = f"[{i+1}/{len(pending)}] {date_iso} TIMEOUT"
                print(msg)
                errors.append(date_iso)
            except Exception as exc:
                msg = f"[{i+1}/{len(pending)}] {date_iso} ERROR: {exc}"
                print(msg)
                errors.append(date_iso)

            # Brief pause to avoid hammering the server
            await page.wait_for_timeout(500)

        await browser.close()

    print(f"\nDone. Downloaded to: {OUTPUT_DIR}")
    if errors:
        print(f"Failed dates ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


def main():
    parser = argparse.ArgumentParser(description="Download NOGA-ISO pie charts for all days.")
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly (may fail reCAPTCHA)")
    parser.add_argument("--from", dest="from_date", default=START_DATE.isoformat(),
                        help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--to", dest="to_date", default=END_DATE.isoformat(),
                        help=f"End date YYYY-MM-DD (default: {END_DATE})")
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    asyncio.run(run(headless=args.headless, from_date=from_date, to_date=to_date))


if __name__ == "__main__":
    main()
