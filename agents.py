import os, json, sqlite3, logging, hashlib
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from dotenv import load_dotenv

# Optional dependencies
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None  # type: ignore
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG ----------
DATA_DIR = os.getenv("DATA_DIR", "marketing_multi_agent_dataset_v1_final")
DB_PATH = os.getenv("AGENT_DB_PATH", "agent_memory.db")

NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "auto").lower()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Neo4jpassword")

REDIS_URL = os.getenv("REDIS_URL")  # e.g. redis://localhost:6379/0

STM_PROMOTE_DAYS = int(os.getenv("STM_PROMOTE_DAYS", "7"))
EPM_MAX          = int(os.getenv("EPM_MAX", "100"))          # global cap
EPM_PER_KEY_MAX  = int(os.getenv("EPM_PER_KEY_MAX", "20"))

MEM_SEED_FROM_CSV = os.getenv("MEM_SEED_FROM_CSV", "false").lower() in {"1", "true", "yes"}
MEM_FORCE_SEED    = os.getenv("MEM_FORCE_SEED", "false").lower() in {"1", "true", "yes"}
MEM_SEED_CAP      = int(os.getenv("MEM_SEED_CAP", "5000"))

# ---------- HELPERS ----------
def _find(*names: str) -> Optional[str]:
    for n in names:
        p1 = os.path.join(DATA_DIR, n)
        if os.path.exists(p1):
            return p1
        if os.path.exists(n):
            return n
    return None

def _read_csv(*names: str, required: bool = True) -> pd.DataFrame:
    p = _find(*names)
    if not p:
        if required:
            raise FileNotFoundError(f"Missing CSV: {', '.join(names)} in '{DATA_DIR}/' or CWD.")
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        if required:
            raise
        logging.warning(f"Failed to read {p}: {e}")
        return pd.DataFrame()

def _to_float(x, default=0.0) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float(default)

def _json_norm(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def _json_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_json_norm(payload).encode("utf-8")).hexdigest()

# ---------- CSVs (required core) ----------
leads_df               = _read_csv("leads.csv")
# Some repos name this file agent_actions.csv, others agents_actions.csv
actions_df             = _read_csv("agent_actions.csv", "agents_actions.csv")
campaigns_df           = _read_csv("campaigns.csv")
campaigns_daily_df     = _read_csv("campaign_daily.csv", "campaigns_daily.csv")

# ---------- CSVs (optional enrichments) ----------
conversations_df           = _read_csv("conversations.csv", required=False)
interactions_df            = _read_csv("interactions.csv", required=False)
ab_variants_df             = _read_csv("ab_variants.csv", required=False)
mcp_jsonrpc_calls_df       = _read_csv("mcp_jsonrpc_calls.csv", required=False)
mcp_resource_access_df     = _read_csv("mcp_resource_access.csv", required=False)
memory_episodic_seed_df    = _read_csv("memory_episodic.csv", required=False)
memory_long_term_seed_df   = _read_csv("memory_long_term.csv", required=False)
memory_short_term_seed_df  = _read_csv("memory_short_term.csv", required=False)
security_auth_events_df    = _read_csv("security_auth_events.csv", required=False)
segments_df                = _read_csv("segments.csv", required=False)
semantic_kg_triples_df     = _read_csv("semantic_kg_triples.csv", required=False)
transport_http_requests_df = _read_csv("transport_http_requests.csv", required=False)
transport_ws_sessions_df   = _read_csv("transport_websocket_sessions.csv", required=False)
conversions_df             = _read_csv("conversions.csv", required=False)

# ---------- SQLite ----------
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("PRAGMA journal_mode=WAL;")
cur.execute("PRAGMA synchronous=NORMAL;")

cur.execute("""
CREATE TABLE IF NOT EXISTS memory (
  agent_id  TEXT,
  type      TEXT,   -- short_term|long_term|episodic
  key       TEXT,   -- lead_id/campaign_id/conv_id
  data      TEXT,   -- JSON
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS memory_dedup (
  agent_id  TEXT,
  type      TEXT,
  key       TEXT,
  hash      TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(agent_id, type, key, hash)
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key)")
conn.commit()

# ---------- Redis (optional) ----------
rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        rds.ping()
        logging.info("Redis cache ENABLED")
    except Exception as e:
        logging.warning(f"Redis disabled: {e}")
        rds = None
else:
    logging.info("Redis cache DISABLED")

# In-proc caches for hot path
_cache: Dict[str, Dict[str, Dict[str, Any]]] = {
    "short_term": {}, "long_term": {}, "episodic": {}
}

def _cache_key(agent_id: str, mem_type: str, key: str) -> str:
    return f"mem:{mem_type}:{agent_id}:{key}"

def cache_get(agent_id: str, mem_type: str, key: str) -> Optional[Dict[str, Any]]:
    ck = _cache_key(agent_id, mem_type, key)
    if rds:
        raw = rds.get(ck)
        if raw:
            return json.loads(raw)
    return _cache.get(mem_type, {}).get(f"{agent_id}:{key}")

def cache_put(agent_id: str, mem_type: str, key: str, payload: Dict[str, Any], ttl: int = 3600):
    ck = _cache_key(agent_id, mem_type, key)
    _cache.setdefault(mem_type, {})[f"{agent_id}:{key}"] = payload
    if rds:
        rds.setex(ck, ttl, _json_norm(payload))

# ---------- Event sink (wired by app.py) ----------
_EVENT_SINK = None
def set_event_sink(fn):  # fn(evt: str, data: dict) -> None
    global _EVENT_SINK
    _EVENT_SINK = fn
    logging.info("Event sink set")

def _emit(evt: str, data: Dict[str, Any]):
    if _EVENT_SINK:
        try:
            _EVENT_SINK(evt, data)
        except Exception as e:
            logging.error(f"Event emission failed: {e}")

# ---------- Neo4j (optional semantic memory) ----------
neo_driver = None
if GraphDatabase is not None and NEO4J_ENABLED != "false":
    try:
        neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with neo_driver.session() as s:
            s.run("RETURN 1").single()
        logging.info("Neo4j ENABLED")
    except Exception as e:
        logging.warning(f"Neo4j disabled: {e}")
        neo_driver = None
else:
    if GraphDatabase is None:
        logging.warning("Neo4j driver not installed; semantic features disabled.")

def add_semantic_triple(subject: str, predicate: str, obj: str, weight: float = 1.0):
    if neo_driver is None:
        return
    try:
        with neo_driver.session() as session:
            session.run("""
                MERGE (s:Entity {name:$s})
                MERGE (o:Entity {name:$o})
                MERGE (s)-[r:REL {predicate:$p}]->(o)
                SET r.weight = coalesce(r.weight,0)+$w, r.updatedAt = datetime()
            """, s=subject, o=obj, p=predicate, w=float(weight))
    except Exception as e:
        logging.error(f"Neo4j triple add failed: {e}")

def query_semantic_for_lead(lead_id: str) -> List[Dict[str, Any]]:
    if neo_driver is None:
        return []
    try:
        with neo_driver.session() as session:
            res = session.run("""
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE s.name CONTAINS $needle
                RETURN s.name AS subject, r.predicate AS predicate, o.name AS object
                LIMIT 10
            """, needle=str(lead_id))
            return [dict(r) for r in res]
    except Exception as e:
        logging.error(f"Neo4j query failed: {e}")
        return []

# ---------- Memory core ----------
def save_memory(agent_id: str, mem_type: str, payload: Dict[str, Any], key: Optional[str] = None) -> bool:
    """Save memory with dedup — returns True if inserted, False if duplicate."""
    if key is None:
        key = str(payload.get("lead_id") or payload.get("campaign_id") or payload.get("conversation_id") or "")
    h = _json_hash(payload)
    try:
        cur.execute(
            "INSERT INTO memory_dedup(agent_id,type,key,hash) VALUES(?,?,?,?)",
            (agent_id, mem_type, key, h),
        )
        cur.execute(
            "INSERT INTO memory(agent_id,type,key,data) VALUES(?,?,?,?)",
            (agent_id, mem_type, key, _json_norm(payload)),
        )
        conn.commit()
        cache_put(agent_id, mem_type, key, payload)
        return True
    except sqlite3.IntegrityError:
        # duplicate payload for same (agent_id,type,key)
        return False

def retrieve_memory(agent_id: str, mem_type: str, key: str) -> Optional[Dict[str, Any]]:
    hit = cache_get(agent_id, mem_type, key)
    if hit:
        return hit
    cur.execute(
        "SELECT data FROM memory WHERE agent_id=? AND type=? AND key=? ORDER BY timestamp DESC LIMIT 1",
        (agent_id, mem_type, key),
    )
    row = cur.fetchone()
    if not row:
        return None
    payload = json.loads(row[0])
    cache_put(agent_id, mem_type, key, payload)
    return payload

def _episodic_for_key(key: str) -> List[Dict[str, Any]]:
    cur.execute(
        "SELECT data FROM memory WHERE type='episodic' AND key=? ORDER BY timestamp DESC LIMIT ?",
        (key, EPM_PER_KEY_MAX * 3),
    )
    return [json.loads(r[0]) for r in cur.fetchall()]

def _summarize_episodes(key: str) -> Dict[str, Any]:
    """Lightweight roll-up: top actions/variants/outcomes frequencies."""
    eps = _episodic_for_key(key)
    if not eps:
        return {"summary": "no episodes"}
    freq: Dict[str, int] = {}
    outcomes: Dict[str, int] = {}
    variants: Dict[str, int] = {}
    for e in eps:
        for a in e.get("action_sequence", []):
            freq[a] = freq.get(a, 0) + 1
        out = str(e.get("outcome") or e.get("decision") or "unknown")
        outcomes[out] = outcomes.get(out, 0) + 1
        vid = str(e.get("variant_id") or "")
        if vid:
            variants[vid] = variants.get(vid, 0) + 1
    top_actions = sorted(freq.items(), key=lambda x: -x[1])[:5]
    top_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])[:5]
    top_variants = sorted(variants.items(), key=lambda x: -x[1])[:5]
    return {
        "key": key,
        "top_actions": top_actions,
        "top_outcomes": top_outcomes,
        "top_variants": top_variants,
        "summary": f"{len(eps)} episodes summarized",
    }

def consolidate_memory():
    """Promote aged STM→LTM; cap episodic (global + per-key) and write summaries."""
    # Promote STM -> LTM
    cur.execute(
        "SELECT rowid, agent_id, key, data FROM memory "
        "WHERE type='short_term' AND timestamp < datetime('now', ?)",
        (f"-{STM_PROMOTE_DAYS} days",),
    )
    rows = cur.fetchall()
    for rowid, agent_id, key, data in rows:
        payload = json.loads(data)
        save_memory(agent_id, "long_term", payload, key)
        cur.execute("DELETE FROM memory WHERE rowid=?", (rowid,))
    conn.commit()

    # Episodic per-key cap
    cur.execute("SELECT key, COUNT(*) FROM memory WHERE type='episodic' GROUP BY key")
    for key, cnt in cur.fetchall():
        if cnt > EPM_PER_KEY_MAX:
            to_trim = cnt - EPM_PER_KEY_MAX
            cur.execute(
                "DELETE FROM memory WHERE rowid IN ("
                " SELECT rowid FROM memory WHERE type='episodic' AND key=? ORDER BY timestamp ASC LIMIT ?)",
                (key, to_trim),
            )
            summary = _summarize_episodes(key)
            save_memory("episodic_rollup", "long_term", summary, key=f"ep:{key}")

    # Global episodic cap
    cur.execute("SELECT COUNT(*) FROM memory WHERE type='episodic'")
    (total_epm,) = cur.fetchone()
    if total_epm and total_epm > EPM_MAX:
        trim = total_epm - EPM_MAX
        cur.execute(
            "DELETE FROM memory WHERE rowid IN ("
            " SELECT rowid FROM memory WHERE type='episodic' ORDER BY timestamp ASC LIMIT ?)",
            (trim,),
        )
    conn.commit()

# ---------- CSV -> SQLite memory seed (optional) ----------
def _seed_mem(df: pd.DataFrame, mem_type: str, agent_id: str = "seed", cap: int = MEM_SEED_CAP) -> int:
    if df.empty:
        return 0
    rows = 0
    for _, r in df.head(cap).iterrows():
        payload = r.to_dict()
        key = str(payload.get("lead_id") or payload.get("campaign_id") or payload.get("conversation_id") or "")
        if not key:
            continue
        inserted = save_memory(agent_id, mem_type, payload, key=key)
        rows += 1 if inserted else 0
    return rows

if MEM_SEED_FROM_CSV:
    cur.execute("SELECT COUNT(*) FROM memory")
    (mem_count,) = cur.fetchone()
    if mem_count == 0 or MEM_FORCE_SEED:
        lt = _seed_mem(memory_long_term_seed_df,  "long_term")
        st = _seed_mem(memory_short_term_seed_df, "short_term")
        ep = _seed_mem(memory_episodic_seed_df,   "episodic")
        logging.info(f"Seeded memory from CSVs → long_term={lt}, short_term={st}, episodic={ep}")
    else:
        logging.info("Skipping CSV memory seed (memory table not empty). Set MEM_FORCE_SEED=true to override.")

# ---------- AGENTS ----------
def triage_lead(lead_id: str):
    lead_id = str(lead_id)
    row = leads_df.loc[leads_df["lead_id"].astype(str) == lead_id]
    if row.empty:
        return {"error": f"lead_id {lead_id} not found"}
    lead = row.iloc[0]

    triage = str(lead.get("triage_category") or "").strip()
    if not triage:
        score = _to_float(lead.get("lead_score"))
        triage = "Campaign Qualified" if score > 80 else "Cold Lead" if score > 30 else "General Inquiry"

    # LTM override if exists
    ltm = retrieve_memory("lead_triage", "long_term", lead_id)
    if ltm and isinstance(ltm, dict):
        triage = ltm.get("category", triage)

    # Semantic hints
    for rel in query_semantic_for_lead(lead_id):
        if rel.get("predicate") == "preferred_channel":
            lead["preferred_channel"] = rel.get("object")

    # Very light risk flag from security_auth_events (if present)
    try:
        risk_flag = False
        if not security_auth_events_df.empty:
            # mark recent auth failures (not keyed by lead_id in all datasets; this is illustrative)
            recent = security_auth_events_df.tail(100)
            if "result" in recent.columns:
                fails = recent["result"].astype(str).str.lower().isin({"expired", "denied", "failed"}).sum()
                risk_flag = bool(fails > 0)
    except Exception:
        risk_flag = False

    result = {
        "lead_id": lead_id,
        "category": triage,
        "lead_score": _to_float(lead.get("lead_score")),
        "source": lead.get("source"),
        "campaign_id": str(lead.get("campaign_id")) if pd.notna(lead.get("campaign_id")) else None,
        "preferred_channel": lead.get("preferred_channel"),
        "gdpr_consent": (str(lead.get("gdpr_consent")).lower() == "true") if pd.notna(lead.get("gdpr_consent")) else None,
        "risk_flag": risk_flag,
    }
    save_memory("lead_triage", "short_term", {"lead_id": lead_id, "category": triage}, key=lead_id)
    save_memory("lead_triage", "episodic", {"episode": f"triage:{lead_id}", "outcome": triage}, key=lead_id)
    _emit("lead_triaged", result)
    return result

def engage_lead(lead_id: str, category: str):
    lead_id = str(lead_id)
    if category != "Campaign Qualified":
        return {"status": "pending", "reason": "Not qualified yet"}

    preferred = "email"
    lr = leads_df.loc[leads_df["lead_id"].astype(str) == lead_id]
    if not lr.empty and pd.notna(lr.iloc[0].get("preferred_channel")):
        preferred = str(lr.iloc[0]["preferred_channel"]).strip().lower()

    # choose action
    def _email_like(df: pd.DataFrame) -> pd.DataFrame:
        if "action_type" not in df.columns:
            return df.iloc[0:0]
        t = df["action_type"].astype(str).str.lower()
        return df[t.isin(["email_campaign", "email", "outreach_email"])]

    per_lead = actions_df[actions_df["lead_id"].astype(str) == lead_id] if "lead_id" in actions_df.columns else actions_df.iloc[0:0]
    cand = _email_like(per_lead)
    action = cand.iloc[0].to_dict() if not cand.empty else (per_lead.iloc[0].to_dict() if not per_lead.empty else {})
    if not action:
        any_email = _email_like(actions_df)
        action = any_email.iloc[0].to_dict() if not any_email.empty else (actions_df.iloc[0].to_dict() if not actions_df.empty else {})

    raw_action_id = action.get("action_id")
    action_id = str(raw_action_id) if raw_action_id is not None else None

    message = "Hi! Thanks for your interest — here’s more about our latest campaign."
    try:
        if not conversations_df.empty and {"lead_id", "message"}.issubset(conversations_df.columns):
            msgs = conversations_df[conversations_df["lead_id"].astype(str) == lead_id]
            if not msgs.empty:
                message = str(msgs.iloc[0]["message"])
    except Exception as e:
        logging.warning(f"Message retrieval failed: {e}")

    ctx = action.get("handoff_context_json")
    try:
        ctx = json.loads(ctx) if isinstance(ctx, str) and ctx.strip() else None
    except Exception:
        ctx = {"raw": ctx}

    a_type = str(action.get("action_type") or "").strip().lower()
    if a_type not in {"email_campaign", "email", "outreach_email"}:
        a_type = "call" if preferred == "call" else "email_campaign"
        if a_type == "call":
            message = f"Call scheduled for lead {lead_id}. Script: {message}"

    result = {
        "lead_id": lead_id,
        "action_type": a_type,
        "action_id": action_id,
        "message": message,
        "context": ctx,
        "status": "sent",
    }

    # enrich from interactions/variants/conversions
    try:
        if not interactions_df.empty and "outcome" in interactions_df.columns:
            ir = interactions_df[interactions_df["lead_id"].astype(str) == lead_id]
            if not ir.empty and pd.notna(ir.iloc[0]["outcome"]):
                result["outcome"] = str(ir.iloc[0]["outcome"])
    except Exception as e:
        logging.warning(f"Interaction lookup failed: {e}")

    try:
        campaign_id = str(lr.iloc[0]["campaign_id"]) if not lr.empty and pd.notna(lr.iloc[0].get("campaign_id")) else None
        if campaign_id and not ab_variants_df.empty and "campaign_id" in ab_variants_df.columns:
            vr = ab_variants_df[ab_variants_df["campaign_id"].astype(str) == campaign_id]
            if not vr.empty and "variant_id" in vr.columns:
                result["variant_id"] = str(vr.iloc[0]["variant_id"])
    except Exception as e:
        logging.warning(f"Variant lookup failed: {e}")

    try:
        if not conversions_df.empty:
            cr = conversions_df[conversions_df["lead_id"].astype(str) == lead_id] if "lead_id" in conversions_df.columns else conversions_df.iloc[0:0]
            if not cr.empty:
                result["converted"] = True
                # optional revenue enrichment
                if "revenue_usd" in cr.columns and pd.notna(cr.iloc[0]["revenue_usd"]):
                    result["conversion_revenue"] = _to_float(cr.iloc[0]["revenue_usd"])
            else:
                result["converted"] = False
    except Exception as e:
        logging.warning(f"Conversion lookup failed: {e}")

    # save memories + SEM
    save_memory("engagement", "short_term", result, key=lead_id)
    save_memory("engagement", "long_term", {"lead_id": lead_id, "preferences": {"channel": preferred}}, key=lead_id)
    save_memory(
        "engagement",
        "episodic",
        {"episode": f"engage:{lead_id}", "action_sequence": [a_type], "variant_id": result.get("variant_id"), "outcome": result.get("status")},
        key=lead_id,
    )
    add_semantic_triple(f"Lead:{lead_id}", "engaged_via", a_type)
    _emit("engagement_sent", result)
    return result

def optimize_campaign(campaign_id: str):
    campaign_id = str(campaign_id)
    row = campaigns_df.loc[campaigns_df["campaign_id"].astype(str) == campaign_id]
    if row.empty:
        return {"error": f"campaign_id {campaign_id} not found"}
    campaign = row.iloc[0]

    kpi = str(campaign.get("kpi", "")).strip().upper()
    metric = "roas" if "ROAS" in kpi else "ctr" if "CTR" in kpi else "conversions" if "CONVERSION" in kpi else "roas"
    threshold = 1.2 if "ROAS" in kpi else 0.02 if "CTR" in kpi else 5.0 if "CONVERSION" in kpi else 1.2

    daily = campaigns_daily_df.loc[campaigns_daily_df["campaign_id"].astype(str) == campaign_id]
    if daily.empty or metric not in daily.columns:
        result = {
            "campaign_id": campaign_id,
            "kpi": kpi or "ROAS",
            "metric": metric,
            "decision": "insufficient_data",
            "reason": f"Missing '{metric}'",
        }
        save_memory("campaign_opt", "short_term", result, key=campaign_id)
        _emit("campaign_decided", result)
        return result

    series = pd.to_numeric(daily[metric], errors="coerce").dropna()
    performance = float(series.mean()) if not series.empty else 0.0
    decision = "continue" if performance >= threshold else "escalate"

    result: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "kpi": kpi or "ROAS",
        "metric": metric,
        "performance": performance,
        "threshold": threshold,
        "decision": decision,
    }

    # HTTP / WS error hints (transport logs)
    try:
        if not transport_http_requests_df.empty and "status_code" in transport_http_requests_df.columns:
            last_http = transport_http_requests_df.tail(500)
            http_errors = (pd.to_numeric(last_http["status_code"], errors="coerce") >= 400).sum()
            result["recent_http_errors"] = int(http_errors)
        if not transport_ws_sessions_df.empty and "messages_received" in transport_ws_sessions_df.columns:
            last_ws = transport_ws_sessions_df.tail(200)
            result["ws_activity"] = int(pd.to_numeric(last_ws["messages_received"], errors="coerce").fillna(0).sum())
    except Exception as e:
        logging.warning(f"Transport enrichment failed: {e}")

    # Resource access logs
    try:
        if not mcp_resource_access_df.empty and "resource_uri" in mcp_resource_access_df.columns:
            hits = mcp_resource_access_df[mcp_resource_access_df["resource_uri"].astype(str).str.contains(campaign_id, na=False)]
            if not hits.empty and "success" in hits.columns:
                val = str(hits.iloc[0]["success"]).lower()
                result["access_success"] = val in {"1", "true", "yes"}
    except Exception as e:
        logging.warning(f"Resource access lookup failed: {e}")

    # Segment rules (first matching example)
    try:
        if not segments_df.empty and "rules_json" in segments_df.columns:
            for _, r in segments_df.iterrows():
                try:
                    rules = json.loads(str(r["rules_json"]))
                except Exception:
                    continue
                if isinstance(rules, dict):
                    result["segment_rules"] = rules
                    break
    except Exception as e:
        logging.warning(f"Segment lookup failed: {e}")

    # Conversion rate (if conversions.csv keyed by campaign_id)
    try:
        if not conversions_df.empty and "campaign_id" in conversions_df.columns:
            c_rows = conversions_df[conversions_df["campaign_id"].astype(str) == campaign_id]
            if not c_rows.empty:
                result["conversions_count"] = int(len(c_rows))
    except Exception as e:
        logging.warning(f"Conversion enrichment failed: {e}")

    save_memory("campaign_opt", "short_term", result, key=campaign_id)
    save_memory("campaign_opt", "episodic", {"episode": f"opt:{campaign_id}", "action_sequence": ["decide"], "decision": decision}, key=campaign_id)
    add_semantic_triple(f"Campaign:{campaign_id}", "optimized_for", result["kpi"])
    _emit("campaign_decided", result)

    # opportunistic consolidation
    consolidate_memory()
    return result

# ---------- JSON-RPC registry ----------
METHODS = {
    "triage_lead": triage_lead,
    "engage_lead": engage_lead,
    "optimize_campaign": optimize_campaign,
}

if __name__ == "__main__":
    print("Run app.py to start the server.")
