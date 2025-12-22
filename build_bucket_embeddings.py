# build_bucket_embeddings.py
import os, json, numpy as np
from openai import OpenAI

from config import FULL_DIR, EMB_DIR, EMBEDDING_MODEL

def _safe_issue_filename(issue: str) -> str:
    s = issue.replace(" ", "_").replace("/", "-")
    return s

def _load_json(path: str):
    if not os.path.exists(path): 
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_texts(client: OpenAI, texts):
    # Batch embedding
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    os.makedirs(EMB_DIR, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for country in os.listdir(FULL_DIR):
        cpath = os.path.join(FULL_DIR, country)
        if not os.path.isdir(cpath): 
            continue
        out_cdir = os.path.join(EMB_DIR, country)
        os.makedirs(out_cdir, exist_ok=True)

        for fname in os.listdir(cpath):
            if not fname.endswith(".json"): 
                continue
            full = _load_json(os.path.join(cpath, fname))
            entries = full.get("entries", [])
            if not entries:
                continue

            texts = []
            meta = []
            for e in entries:
                t = " | ".join([
                    e.get("position_summary",""),
                    e.get("quote",""),
                    f'{e.get("source_file","")} p.{e.get("page")} ¶{e.get("paragraph_id")}'
                ])
                texts.append(t)
                meta.append({
                    "source_file": e["source_file"],
                    "page": e["page"],
                    "paragraph_id": e["paragraph_id"],
                    "confidence": e.get("confidence"),
                    "weight": e.get("weight")
                })

            embs = embed_texts(client, texts)
            arr = np.array(embs, dtype=np.float32)
            np.savez_compressed(
                os.path.join(out_cdir, fname.replace(".json", ".embeddings.npz")),
                embeddings=arr, meta=np.array(meta, dtype=object)
            )
    print("Embedding indexes built at", EMB_DIR)

if __name__ == "__main__":
    main()
