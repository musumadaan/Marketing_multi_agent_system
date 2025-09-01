#!/usr/bin/env python3
import os, json, sqlite3, argparse
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "marketing_multi_agent_dataset_v1_final")
DB_PATH  = os.getenv("AGENT_DB_PATH", "agent_memory.db")

def path(*names):
    for n in names:
        p1 = os.path.join(DATA_DIR, n)
        if os.path.exists(p1): return p1
        if os.path.exists(n):  return n
    return None

def read_csv(*names, required=False):
    p = path(*names)
    if not p:
        if required: raise FileNotFoundError(f"Missing CSV: {names}")
        return pd.DataFrame()
    return pd.read_csv(p)

def main():
    ap = argparse.ArgumentParser(description="Seed SQLite with memory & interactions")
    ap.add_argument("--reset", action="store_true", help="drop & recreate tables")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    if args.reset:
        cur.execute("DROP TABLE IF EXISTS memory")
        cur.execute("DROP TABLE IF EXISTS memory_dedup")
        cur.execute("DROP TABLE IF EXISTS interactions")
        conn.commit()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory (
      agent_id  TEXT,
      type      TEXT,
      key       TEXT,
      data      TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory_dedup (
      agent_id  TEXT,
      type      TEXT,
      key       TEXT,
      hash      TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(agent_id, type, key, hash)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
      event_type TEXT,
      lead_id TEXT,
      campaign_id TEXT,
      payload TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()

    # Add payload column if missing (safe no-op if present)
    try:
        cur.execute("SELECT payload FROM interactions LIMIT 1;")
    except sqlite3.OperationalError:
        cur.execute("ALTER TABLE interactions ADD COLUMN payload TEXT;")
        conn.commit()

    # ------- Seed memory from CSVs -------
    st = read_csv("memory_short_term.csv")
    lt = read_csv("memory_long_term.csv")
    ep = read_csv("memory_episodic.csv")

    def upsert_mem(df: pd.DataFrame, mem_type: str):
        if df.empty: return
        for _, r in df.iterrows():
            data = r.to_dict()
            key = str(data.get("lead_id") or data.get("campaign_id") or data.get("conversation_id") or "")
            cur.execute("INSERT INTO memory(agent_id,type,key,data) VALUES(?,?,?,?)",
                        ("seed", mem_type, key, json.dumps(data)))
        print(f"[ok] seeded {mem_type}: {len(df)} rows")

    upsert_mem(st, "short_term")
    upsert_mem(lt, "long_term")
    upsert_mem(ep, "episodic")
    conn.commit()

    # ------- Seed interactions from interactions.csv (optional) -------
    inter = read_csv("interactions.csv")
    if not inter.empty:
        cols = {c.lower(): c for c in inter.columns}
        for _, r in inter.iterrows():
            event_type = str(r.get(cols.get("event_type"), "") or "")
            lead_id    = str(r.get(cols.get("lead_id"), "") or "")
            campaign_id= str(r.get(cols.get("campaign_id"), "") or "")
            # pack the rest into payload
            payload = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
            cur.execute("INSERT INTO interactions(event_type, lead_id, campaign_id, payload) VALUES(?,?,?,?)",
                        (event_type, lead_id, campaign_id, json.dumps(payload)))
        print(f"[ok] seeded interactions: {len(inter)} rows")
        conn.commit()

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_key ON memory(key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inter_ts ON interactions(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inter_lc ON interactions(lead_id, campaign_id)")
    conn.commit()
    conn.close()
    print("Ingest complete.")

if __name__ == "__main__":
    main()
