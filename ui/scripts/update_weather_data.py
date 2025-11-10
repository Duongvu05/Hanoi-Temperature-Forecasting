
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import tempfile
import shutil
from io import StringIO  
# -------------------------- CONFIG --------------------------
API_KEY = "PT84G2MR6JNCGJRPNDMJ89XQW"

CITY = "Hanoi"
CHUNK_DAYS = 365
RETRY_ATTEMPTS = 3
SLEEP_BETWEEN_CHUNKS = 1

BASE_DIR = Path(__file__).parent.parent.parent
EXISTING_DATA_PATH = BASE_DIR / "ui" / "data" / "hanoi_weather_complete.csv"
MODEL_DATA = BASE_DIR / "data" / "raw" / "daily_data.csv"

def load_existing() -> pd.DataFrame:
    if EXISTING_DATA_PATH.exists():
        df = pd.read_csv(EXISTING_DATA_PATH, parse_dates=["datetime"])
    else:
        print(f"{EXISTING_DATA_PATH} not found. Loading from model data: {MODEL_DATA}")
        df = pd.read_csv(MODEL_DATA, parse_dates=["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} rows – last date: {df['datetime'].max().date()}")
    return df


def build_date_chunks(start: datetime, end: datetime) -> List[tuple]:
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{CITY}/{start}/{end}?"
        f"unitGroup=metric&include=days&key={API_KEY}&contentType=csv"  # ← CHANGED TO CSV
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                # Parse CSV directly from text response
                df = pd.read_csv(StringIO(r.text))
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            else:
                print(f"Attempt {attempt+1}/{RETRY_ATTEMPTS} failed ({r.status_code}) – retrying...")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Network error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {start} to {end}")


def safe_save_csv(df: pd.DataFrame, path: Path, max_retries: int = 5) -> None:
    temp_file = path.with_suffix(".tmp")
    for attempt in range(max_retries):
        try:
            df.to_csv(temp_file, index=False)
            shutil.move(temp_file, path)
            print(f"Saved to {path}")
            return
        except PermissionError:
            print(f"File in use. Retrying in 2s... ({attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"Save error: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to save {path}")


def main() -> None:
    df_raw = load_existing()
    last_raw_date = df_raw["datetime"].max().date()
    today = datetime.now().date()  # ← CHANGED: get TODAY

    if last_raw_date >= today:
        print("Dataset already up-to-date – nothing to do.")
        return

    start_date = last_raw_date + timedelta(days=1)
    print(f"Need data from {start_date} to {today}")  # ← includes today

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(today, datetime.min.time())  # ← up to today

    chunks = build_date_chunks(start_dt, end_dt)
    new_frames: List[pd.DataFrame] = []

    for i, (s, e) in enumerate(chunks, 1):
        print(f"Chunk {i}/{len(chunks)}: {s} - {e}")
        chunk_df = fetch_chunk(s, e)
        print(f" {len(chunk_df)} days received")
        new_frames.append(chunk_df)
        time.sleep(SLEEP_BETWEEN_CHUNKS)

    new_df = pd.concat(new_frames, ignore_index=True)
    print(f"Total new days fetched: {len(new_df)}")

    # Align columns
    common_cols = df_raw.columns.intersection(new_df.columns)
    df_raw = df_raw[common_cols]
    new_df = new_df[common_cols]

    # Merge & dedupe
    full = pd.concat([df_raw, new_df], ignore_index=True)
    full = full.drop_duplicates(subset="datetime", keep="last")
    full = full.sort_values("datetime").reset_index(drop=True)

    safe_save_csv(full, EXISTING_DATA_PATH)

    print("\nSUCCESS!")
    print(f"   Total rows : {len(full)}")
    print(f"   Date range : {full['datetime'].min().date()} to {full['datetime'].max().date()}")
    print(f"   Updated    : {EXISTING_DATA_PATH}")

if __name__ == "__main__":
    main()