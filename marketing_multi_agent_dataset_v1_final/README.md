# Marketing Multi‑Agent System — Synthetic Dataset (v1)

This bundle is designed for AIML engineer assessments around a 3‑agent marketing system:
Lead Triage, Engagement, and Campaign Optimization. It includes leads, campaigns, interactions,
agent actions (handoffs/escalations), daily KPIs, memory artifacts (short/long/episodic/semantic),
and MCP/JSON‑RPC transport logs for server/client evaluation.

**Time window:** 2025-05-01 to 2025-08-29

Files:
- `campaigns.csv` — campaign metadata and targets
- `ab_variants.csv` — A/B creative variants
- `leads.csv` — lead records with triage category and score
- `interactions.csv` — event‑level engagement logs (email, SMS, calls, social, site)
- `conversations.csv` — conversation sessions per lead
- `agent_actions.csv` — triage/outreach/optimize/handoff/escalate events with context
- `campaign_daily.csv` — daily KPIs (impressions, clicks, conversions, ROAS, etc.)
- `conversions.csv` — lead‑level conversion outcomes
- `memory_short_term.csv` — per‑conversation ephemeral context
- `memory_long_term.csv` — durable lead preferences
- `memory_episodic.csv` — successful playbooks with outcomes
- `semantic_kg_triples.csv` — domain knowledge graph edges
- `mcp_jsonrpc_calls.csv` — JSON‑RPC 2.0 call logs across HTTP/WebSocket
- `transport_websocket_sessions.csv` — WebSocket session metrics
- `transport_http_requests.csv` — HTTP request metrics
- `mcp_resource_access.csv` — resource access ledger for MCP
- `segments.csv` — cohort definitions with rule JSON
- `security_auth_events.csv` — auth events for production hardening analysis
- `data_dictionary.md` — column‑level descriptions

Suggested uses:
- Lead triage classification; outreach sequencing; A/B analysis; attribution
- Agent handoff protocol evaluation; escalation policy testing
- Memory consolidation & retrieval experiments; KG reasoning
- MCP/transport reliability dashboards; SLO and security posture analysis
