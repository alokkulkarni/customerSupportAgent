# type: ignore
import os
import shutil
import sqlite3
import requests

import pandas as pd


db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"

backup_file = "travel2.backup.sqlite"
overwrite = False

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    with open(local_file, "wb") as f:
        f.write(response.content)
    shutil.copy(local_file, backup_file)

def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql("SELECT * FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {}
    
    for table in tables:
        tdf[table] = pd.read_sql(f"SELECT * FROM {table}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    
    del df
    
    del tdf
    
    conn.commit()
    
    conn.close()

    return file

db = update_dates(local_file)
