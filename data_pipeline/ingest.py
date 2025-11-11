from pathlib import Path
import json, hashlib, shutil, time
from datetime import datetime

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
DOCS = Path("docs"); DOCS.mkdir(parents=True, exist_ok=True)

def save_telemetry(batch: list[dict]):
    """
    Each item example:
    {"shipment_id":"SHIP-001","ts":"2025-11-10T18:00:00Z",
     "temp_voted":70.2,"pressure_voted":2.03,
     "T_low":60,"T_high":90,"T_sis_low":55,"T_sis_high":95,
     "P_low":1.5,"P_high":2.5,"P_sis_low":1.2,"P_sis_high":2.8,
     "anomaly_flag":0}
    """
    ts = int(time.time())
    out = RAW / f"telemetry_{ts}.json"
    out.write_text(json.dumps(batch, ensure_ascii=False))
    return str(out)

def index_document(policy_id: str, src_path: str, doc_type: str):
    """Copy a policy/claim doc and append an index line in docs/doc_index.jsonl."""
    src = Path(src_path)
    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dst_dir = DOCS / doc_type; dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{policy_id}_{sha[:12]}_{src.name}"
    shutil.copy2(src, dst)
    idx = {
        "doc_id": sha, "policy_id": policy_id, "doc_type": doc_type,
        "filename": dst.name, "sha256": sha,
        "received_at": datetime.utcnow().isoformat(), "path": str(dst)
    }
    with open("docs/doc_index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(idx) + "\n")
    return str(dst)

if __name__ == "__main__":
    # Demo batch so you can test quickly
    demo = [{
        "shipment_id": "SHIP-001",
        "ts": datetime.utcnow().isoformat() + "Z",
        "temp_voted": 70.0, "pressure_voted": 2.02,
        "T_low": 60, "T_high": 90, "T_sis_low": 55, "T_sis_high": 95,
        "P_low": 1.5, "P_high": 2.5, "P_sis_low": 1.2, "P_sis_high": 2.8,
        "anomaly_flag": 0
    }]
    print("Wrote:", save_telemetry(demo))
