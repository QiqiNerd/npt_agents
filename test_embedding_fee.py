# estimate_embedding_costs.py
import os, json, tiktoken

FULL_DIR = "outputs/agent_buckets"
ENC = tiktoken.get_encoding("cl100k_base")
PRICE_PER_1K = {
    "text-embedding-3-large": 0.00013,  # $ per 1K tokens
    "text-embedding-3-small": 0.00002,
}

def toks(s: str) -> int:
    return len(ENC.encode(s or ""))

def main():
    total_tokens = 0
    files = 0
    entries_cnt = 0

    for country in os.listdir(FULL_DIR):
        cpath = os.path.join(FULL_DIR, country)
        if not os.path.isdir(cpath): 
            continue
        for fname in os.listdir(cpath):
            if not fname.endswith(".json"): 
                continue
            path = os.path.join(cpath, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            if not entries: 
                continue
            files += 1
            entries_cnt += len(entries)
            for e in entries:
                t = " | ".join([
                    e.get("position_summary",""),
                    e.get("quote",""),
                    f'{e.get("source_file","")} p.{e.get("page")} ¶{e.get("paragraph_id")}'
                ])
                total_tokens += toks(t)

    print(f"Files scanned: {files}, entries: {entries_cnt}")
    print(f"Total tokens (cl100k_base): {total_tokens:,}")
    for model, price in PRICE_PER_1K.items():
        usd = (total_tokens / 1000.0) * price
        print(f"{model}: ~${usd:.2f} USD")

if __name__ == "__main__":
    main()
