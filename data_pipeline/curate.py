import duckdb, pandas as pd, pathlib

RAW = pathlib.Path("data/raw")
SILVER = pathlib.Path("data/silver")
SILVER.mkdir(parents=True, exist_ok=True)

def bronze_to_silver():
    files = sorted(RAW.glob("telemetry_*.json"))
    if not files:
        print("No raw files found."); return None
    frames = [pd.read_json(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["shipment_id","ts"])

    # derive breaches
    df["breach_normal"] = ((df["temp_voted"] < df["T_low"]) | (df["temp_voted"] > df["T_high"]) |
                           (df["pressure_voted"] < df["P_low"]) | (df["pressure_voted"] > df["P_high"]))
    df["breach_sis"] = ((df["temp_voted"] < df["T_sis_low"]) | (df["temp_voted"] > df["T_sis_high"]) |
                        (df["pressure_voted"] < df["P_sis_low"]) | (df["pressure_voted"] > df["P_sis_high"]))

    out = SILVER / "telemetry.parquet"
    df.to_parquet(out, index=False)
    print("Silver written:", out)
    return str(out)

def load_to_duckdb(parquet_path="data/silver/telemetry.parquet", db_path="warehouse.duckdb"):
    con = duckdb.connect(db_path)
    con.execute(open("schemas.sql","r",encoding="utf-8").read())
    con.execute("""
        CREATE OR REPLACE TABLE fact_telemetry AS
        SELECT shipment_id, ts, temp_voted, pressure_voted,
               CAST(COALESCE(anomaly_flag,0) AS INTEGER) AS anomaly_flag,
               CAST(breach_normal AS INTEGER) AS breach_normal,
               CAST(breach_sis AS INTEGER) AS breach_sis
        FROM parquet_scan(?);
    """, [parquet_path])
    con.close()
    print("Loaded into:", db_path)

if __name__ == "__main__":
    pq = bronze_to_silver()
    if pq: load_to_duckdb(pq)
