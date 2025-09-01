# app.py
import os
import json
import logging
import sqlite3
from typing import Any, Dict, Iterable, Tuple, Union

from flask import (
    Flask,
    jsonify,
    request as flask_request,
    send_from_directory,
    redirect,
    url_for,
    make_response,
)
from flask_socketio import SocketIO
from dotenv import load_dotenv

# jsonrpcserver v3.x
from jsonrpcserver import dispatch

# your agents module (must expose METHODS, set_event_sink, add_semantic_triple, query_semantic_for_lead, neo_driver)
import agents

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_SERVER_PORT", "5000"))

DB_PATH = os.getenv("AGENT_DB_PATH", "agent_memory.db")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
OPENAPI_FILE = os.path.join(os.path.dirname(__file__), "openapi.yaml")

# Optional flags
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() in {"1", "true", "yes"}

# -----------------------------------------------------------------------------
# Flask & Socket.IO
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")  # dev-friendly

log.info("Redis cache %s", "ENABLED" if REDIS_ENABLED else "DISABLED")
log.info("Neo4j %s", "ENABLED" if getattr(agents, "neo_driver", None) is not None else "DISABLED")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _json_response(obj: Union[dict, list, str], status: int = 200):
    """Return obj as JSON. If `obj` is already a JSON string, pass it through."""
    if isinstance(obj, str):
        resp = make_response(obj, status)
        resp.headers["Content-Type"] = "application/json"
        return resp
    return make_response(json.dumps(obj, default=str), status, {"Content-Type": "application/json"})

def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=5)

def _ensure_interactions_table(cur: sqlite3.Cursor):
    # Create table with superset schema (won't add columns if already exist, but OK for new DBs)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions(
          event_type   TEXT,
          lead_id      TEXT,
          campaign_id  TEXT,
          payload      TEXT,
          timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

def _columns(cur: sqlite3.Cursor, table: str) -> Iterable[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def _event_sink(evt: str, data: Dict[str, Any]):
    """
    Persist interaction if possible (schema-agnostic), then emit to Socket.IO.
    This is registered into agents.set_event_sink so agent functions call it.
    """
    # Persist
    try:
        conn = _get_conn()
        cur = conn.cursor()
        _ensure_interactions_table(cur)  # ensures table exists (may already exist with narrower schema)

        cols = set(_columns(cur, "interactions"))
        to_write = {
            "event_type": evt,
            "lead_id": str(data.get("lead_id") or ""),
            "campaign_id": str(data.get("campaign_id") or ""),
            "payload": json.dumps(data),
        }
        use_cols = [c for c in ("event_type", "lead_id", "campaign_id", "payload") if c in cols]
        placeholders = ",".join(["?"] * len(use_cols))
        sql = f"INSERT INTO interactions ({','.join(use_cols)}) VALUES ({placeholders})"
        cur.execute(sql, [to_write[c] for c in use_cols])
        conn.commit()
        conn.close()
    except Exception as e:
        app.logger.warning(f"interaction persist failed: {e}")

    # Emit (Flask-SocketIO 5.1.2 has no 'broadcast' kw in this context)
    try:
        socketio.emit(evt, data)
    except Exception as e:
        app.logger.warning(f"socket emit failed: {e}")

# Register event sink so agents.py can call _emit(...)
agents.set_event_sink(_event_sink)

# -----------------------------------------------------------------------------
# Routes: basic pages
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return redirect(url_for("events_page"))

@app.get("/events")
def events_page():
    # Serve the Events UI
    return send_from_directory(STATIC_DIR, "events.html")

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})

# -----------------------------------------------------------------------------
# JSON-RPC endpoint
# -----------------------------------------------------------------------------
@app.post("/mcp")
def mcp_server():
    """
    Accept raw JSON-RPC 2.0 requests and dispatch against agents.METHODS.
    Compatible with jsonrpcserver 3.x.
    """
    try:
        payload = flask_request.get_json(force=True, silent=False)
    except Exception as e:
        return _json_response({"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": None}, 400)

    if not isinstance(payload, dict) or "method" not in payload:
        return _json_response({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}, 400)

    try:
        # dispatch returns a Response or JSON string in v3
        resp = dispatch(payload, methods=agents.METHODS)
        return _json_response(resp, 200)
    except Exception as e:
        return _json_response({"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {e}"}, "id": payload.get("id")}, 500)

# -----------------------------------------------------------------------------
# Backfill: recent interactions (schema-agnostic)
# -----------------------------------------------------------------------------
@app.get("/api/interactions/recent")
def recent_interactions():
    limit = int(flask_request.args.get("limit", "50"))
    limit = max(1, min(500, limit))
    conn = _get_conn()
    cur = conn.cursor()

    # If table doesn't exist, return empty list
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'")
    if not cur.fetchone():
        conn.close()
        return jsonify([])

    # Discover columns dynamically
    cols = list(_columns(cur, "interactions"))
    col_list = ", ".join(cols) if cols else "event_type, lead_id, campaign_id, timestamp"
    try:
        cur.execute(f"SELECT {col_list} FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    except Exception:
        # Fallback to minimal set if timestamp not present
        try:
            cur.execute("SELECT event_type, lead_id, campaign_id FROM interactions LIMIT ?", (limit,))
            rows = cur.fetchall()
            cols = ["event_type", "lead_id", "campaign_id"]
        except Exception:
            conn.close()
            return jsonify([])
    conn.close()

    # Normalize into list of dicts
    out = []
    for row in rows:
        rec = dict(zip(cols, row))
        rec.setdefault("event_type", rec.get("evt") or rec.get("event") or "interaction")
        rec.setdefault("timestamp", rec.get("ts") or rec.get("created_at") or rec.get("time"))
        out.append(rec)
    return jsonify(out)

# -----------------------------------------------------------------------------
# Neo4j utilities
# -----------------------------------------------------------------------------
@app.get("/api/neo4j/ping")
def neo_ping():
    drv = getattr(agents, "neo_driver", None)
    if drv is None:
        return jsonify({"ok": False, "enabled": False, "reason": "driver not installed or disabled"})
    try:
        version = None
        with drv.session() as s:
            try:
                rec = s.run("CALL dbms.components() YIELD name, versions, edition RETURN versions[0] AS version").single()
                if rec:
                    version = rec.get("version")
            except Exception:
                # Neo4j 5 might restrict this; a simple ping is fine
                s.run("RETURN 1").single()
        return jsonify({"ok": True, "enabled": True, "version": version})
    except Exception as e:
        return jsonify({"ok": False, "enabled": True, "reason": str(e)}), 500

@app.post("/api/neo4j/demo")
def neo_demo():
    """Create a sample semantic edge for quick validation and echo a preview."""
    data = flask_request.get_json(force=True, silent=True) or {}
    lead_id = str(data.get("lead_id") or "").strip()
    predicate = str(data.get("predicate") or "engaged_via").strip()
    obj = str(data.get("object") or "Email").strip()

    if not lead_id:
        return jsonify(ok=False, error="lead_id required"), 400

    drv = getattr(agents, "neo_driver", None)
    if drv is None:
        return jsonify(ok=False, error="Neo4j disabled or driver not installed"), 400

    try:
        agents.add_semantic_triple(f"Lead:{lead_id}", predicate, obj, 1.0)
        preview = agents.query_semantic_for_lead(lead_id)
        payload = {"lead_id": lead_id, "predicate": predicate, "object": obj, "preview": preview}

        # live + persist
        try:
            socketio.emit("neo4j_demo", payload)  # no 'broadcast' kw in 5.1.2
            _event_sink("neo4j_demo", payload)    # also store in interactions for backfill
        except Exception:
            pass

        return jsonify(ok=True, lead_id=lead_id, predicate=predicate, object=obj, preview=preview)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

# -----------------------------------------------------------------------------
# Demo helpers (HTTP wrappers around agent functions)
# -----------------------------------------------------------------------------
@app.post("/api/demo/triage")
def api_demo_triage():
    data = flask_request.get_json(force=True, silent=True) or {}
    lead_id = str(data.get("lead_id") or "").strip()
    if not lead_id:
        return jsonify({"error": "lead_id required"}), 400
    res = agents.triage_lead(lead_id=lead_id)
    return jsonify(res)

@app.post("/api/demo/engage")
def api_demo_engage():
    data = flask_request.get_json(force=True, silent=True) or {}
    lead_id = str(data.get("lead_id") or "").strip()
    category = str(data.get("category") or "Campaign Qualified")
    if not lead_id:
        return jsonify({"error": "lead_id required"}), 400
    res = agents.engage_lead(lead_id=lead_id, category=category)
    return jsonify(res)

@app.post("/api/demo/optimize")
def api_demo_optimize():
    data = flask_request.get_json(force=True, silent=True) or {}
    campaign_id = str(data.get("campaign_id") or "").strip()
    if not campaign_id:
        return jsonify({"error": "campaign_id required"}), 400
    res = agents.optimize_campaign(campaign_id=campaign_id)
    return jsonify(res)

# -----------------------------------------------------------------------------
# OpenAPI docs
# -----------------------------------------------------------------------------
@app.get("/openapi.yaml")
def serve_openapi_yaml():
    # serve the YAML file if it exists
    directory = os.path.dirname(OPENAPI_FILE)
    filename = os.path.basename(OPENAPI_FILE)
    if not os.path.exists(OPENAPI_FILE):
        return jsonify({"error": "openapi.yaml not found"}), 404
    return send_from_directory(directory, filename, mimetype="text/yaml")

@app.get("/docs")
def docs():
    # Minimal Redoc page
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>API Docs</title>
    <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
    <style>body,html,#redoc{{height:100%;margin:0;padding:0;background:#0b0f17}} .rd-md-code{{background:#0f1722}}</style>
  </head>
  <body>
    <redoc id="redoc" spec-url="/openapi.yaml"></redoc>
  </body>
</html>
"""
    return make_response(html, 200, {"Content-Type": "text/html"})

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"Event sink set")
    log.info(f"Starting MCP server (Socket.IO): http://{HOST}:{PORT}/mcp")
    # Important: Werkzeug dev server can't handle websockets reliably; our UI uses polling only.
    socketio.run(app, host=HOST, port=PORT, debug=True)

