CREATE TABLE IF NOT EXISTS dim_policy (
  policy_id TEXT PRIMARY KEY,
  insured TEXT, underwriter TEXT,
  commodity TEXT, route TEXT,
  effective_date DATE, expiration_date DATE, coverage_limits REAL
);

CREATE TABLE IF NOT EXISTS dim_shipment (
  shipment_id TEXT PRIMARY KEY,
  policy_id TEXT, container_id TEXT,
  origin TEXT, destination TEXT,
  departure_ts TIMESTAMP, arrival_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_sensor (
  sensor_id TEXT PRIMARY KEY, type TEXT, make_model TEXT, calibration_date DATE
);

CREATE TABLE IF NOT EXISTS fact_telemetry (
  shipment_id TEXT, ts TIMESTAMP,
  temp_voted REAL, pressure_voted REAL,
  anomaly_flag INTEGER, breach_normal INTEGER, breach_sis INTEGER
);

CREATE TABLE IF NOT EXISTS fact_claim (
  claim_id TEXT PRIMARY KEY, policy_id TEXT,
  loss_ts TIMESTAMP, loss_cause TEXT, paid REAL, reserve REAL, status TEXT
);

CREATE TABLE IF NOT EXISTS doc_index (
  doc_id TEXT PRIMARY KEY, policy_id TEXT, doc_type TEXT,
  filename TEXT, sha256 TEXT, received_at TIMESTAMP, path TEXT
);
