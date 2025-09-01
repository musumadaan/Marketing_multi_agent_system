#!/usr/bin/env python3
import os, sys, json, argparse, requests

HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_SERVER_PORT", "5000"))
BASE = f"http://{HOST}:{PORT}"
MCP  = f"{BASE}/mcp"

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def rpc(method, params=None, _id=1):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": _id}
    r = requests.post(MCP, json=body, timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    p = argparse.ArgumentParser(description="MCP demo client")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    t = sub.add_parser("triage");     t.add_argument("--lead-id", required=True)
    e = sub.add_parser("engage");     e.add_argument("--lead-id", required=True); e.add_argument("--category", default="Campaign Qualified")
    o = sub.add_parser("optimize");   o.add_argument("--campaign-id", required=True)

    r = sub.add_parser("recent");     r.add_argument("--limit", type=int, default=50)

    sub.add_parser("neo4j-ping")
    ne = sub.add_parser("neo4j-edge"); ne.add_argument("--lead-id", default="L0000001"); ne.add_argument("--channel", default="Email")

    # Convenience demos that hit the REST helpers (same as UI buttons)
    dt = sub.add_parser("demo-triage");   dt.add_argument("--lead-id", default="L0000001")
    de = sub.add_parser("demo-engage");   de.add_argument("--lead-id", default="L0000001"); de.add_argument("--category", default="Campaign Qualified")
    do = sub.add_parser("demo-optimize"); do.add_argument("--campaign-id", default="CMP00002")

    args = p.parse_args()

    if args.cmd == "health":
        pretty(requests.get(f"{BASE}/healthz", timeout=5).json()); return

    if args.cmd == "triage":
        pretty(rpc("triage_lead", {"lead_id": args.lead_id})); return

    if args.cmd == "engage":
        pretty(rpc("engage_lead", {"lead_id": args.lead_id, "category": args.category})); return

    if args.cmd == "optimize":
        pretty(rpc("optimize_campaign", {"campaign_id": args.campaign_id})); return

    if args.cmd == "recent":
        pretty(requests.get(f"{BASE}/api/interactions/recent", params={"limit": args.limit}, timeout=10).json()); return

    if args.cmd == "neo4j-ping":
        pretty(requests.get(f"{BASE}/api/neo4j/ping", timeout=10).json()); return

    if args.cmd == "neo4j-edge":
        pretty(requests.post(f"{BASE}/api/neo4j/demo-edge", json={"lead_id": args.lead_id, "channel": args.channel}, timeout=10).json()); return

    if args.cmd == "demo-triage":
        pretty(requests.post(f"{BASE}/api/demo/triage", json={"lead_id": args.lead_id}, timeout=10).json()); return

    if args.cmd == "demo-engage":
        pretty(requests.post(f"{BASE}/api/demo/engage", json={"lead_id": args.lead_id, "category": args.category}, timeout=10).json()); return

    if args.cmd == "demo-optimize":
        pretty(requests.post(f"{BASE}/api/demo/optimize", json={"campaign_id": args.campaign_id}, timeout=10).json()); return

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"[HTTP {e.response.status_code}] {e.response.text}", file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)
