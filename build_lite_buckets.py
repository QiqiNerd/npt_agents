# build_lite_buckets.py
import os, json
from datetime import datetime

from config import FULL_DIR, LITE_DIR, LITE_TOP_N

def main():
    os.makedirs(LITE_DIR, exist_ok=True)
    total = 0
    for country in os.listdir(FULL_DIR):
        cpath = os.path.join(FULL_DIR, country)
        if not os.path.isdir(cpath): 
            continue
        out_cdir = os.path.join(LITE_DIR, country)
        os.makedirs(out_cdir, exist_ok=True)
        for fname in os.listdir(cpath):
            if not fname.endswith(".json"): 
                continue
            with open(os.path.join(cpath, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])[:LITE_TOP_N]
            lite = {
                "country": data.get("country"),
                "issue": data.get("issue"),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "entries": [{
                    "source_file": e["source_file"],
                    "page": e["page"],
                    "paragraph_id": e["paragraph_id"],
                    "position_summary": e.get("position_summary",""),
                    "quote": e.get("quote",""),
                    "confidence": e.get("confidence"),
                    "weight": e.get("weight"),
                    "tier": e.get("tier")
                } for e in entries]
            }
            out_path = os.path.join(out_cdir, fname.replace(".json", ".lite.json"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(lite, f, ensure_ascii=False, indent=2)
            total += 1
    print(f"Built {total} lite files at {LITE_DIR}")

if __name__ == "__main__":
    main()
