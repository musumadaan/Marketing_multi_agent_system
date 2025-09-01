
# Data Dictionary

## campaigns.csv
- campaign_id: unique id
- name: human label
- objective: campaign goal (Lead Gen, Awareness, etc.)
- start_date, end_date: ISO 8601
- channel_mix: JSON array of channels used
- daily_budget_usd, total_budget_usd: numeric
- owner_email: manager contact
- primary_region: geo focus
- target_personas: JSON array
- kpi: primary KPI

## ab_variants.csv
- variant_id: unique id for creative
- campaign_id: FK → campaigns
- channel: channel of variant
- creative_type: SubjectLine/AdCopy/LandingPage/SMSCopy
- subject_line, call_to_action, tone, length_words

## leads.csv
- lead_id: unique id
- created_at, last_active_at: ISO 8601
- source: acquisition source
- campaign_id: FK → campaigns
- triage_category: Campaign Qualified / Cold Lead / General Inquiry
- lead_status: New/Open/Qualified/Unqualified/Converted
- lead_score: 0–100
- company_size, industry, persona, region
- preferred_channel, gdpr_consent
- email, phone
- assigned_engagement_agent

## interactions.csv
- interaction_id: unique event id
- conversation_id: FK → conversations
- lead_id: FK → leads
- campaign_id: FK → campaigns
- timestamp: ISO 8601
- channel: Email/SMS/Social/Ads/Web/Call
- event_type: email_sent/open/click/reply/... etc.
- agent_id: engagement agent performing action
- variant_id: FK → ab_variants
- outcome: none/positive/negative/unsubscribe/callback_requested
- metadata_json: JSON of extra properties

## conversations.csv
- conversation_id: unique id
- lead_id: FK → leads
- opened_at, last_event_at: ISO 8601
- status: active/idle/closed

## agent_actions.csv
- action_id: unique id
- timestamp: ISO 8601
- conversation_id, lead_id: FKs
- action_type: triage/outreach/optimize/handoff/escalate/etc.
- source_agent: id of agent
- source_agent_type: LeadTriage/Engagement/Optimizer
- dest_agent_type: LeadTriage/Engagement/Optimizer/Manager
- handoff_context_json: JSON preserving state
- escalation_reason: none/high_value/complaint/legal/complex_request

## campaign_daily.csv
- campaign_id: FK → campaigns
- date: day
- impressions, clicks, ctr
- leads_created, conversions
- cost_usd, revenue_usd, cpl_usd, roas

## conversions.csv
- lead_id: FK → leads
- campaign_id: FK → campaigns
- converted_at: ISO 8601
- conversion_value_usd: numeric
- conversion_type: demo_booked/trial_started/purchase/meeting_scheduled

## memory_short_term.csv
- conversation_id: FK → conversations
- lead_id: FK → leads
- last_utterance_summary: text
- active_intent: canonical intent
- slots_json: JSON of extracted fields
- expires_at: ISO 8601 (TTL)

## memory_long_term.csv
- lead_id: FK → leads
- region, industry
- rfm_score: 0–1
- preferences_json: JSON of preferences (contact time, channels, interests)
- last_updated_at: ISO 8601

## memory_episodic.csv
- episode_id: unique
- scenario: situation label
- action_sequence_json: JSON array of steps
- outcome_score: 0–1
- notes: free text

## semantic_kg_triples.csv
- subject, predicate, object: KG triple
- weight: edge weight 0.1–1.0
- source: provenance

## mcp_jsonrpc_calls.csv
- rpc_id: unique
- timestamp: ISO 8601
- transport: WebSocket/HTTP
- method: JSON‑RPC method name
- params_bytes: payload size
- duration_ms: latency
- status_code: response status
- source_agent_type: actor
- target_resource: e.g., db://leads, kg://graph

## transport_websocket_sessions.csv
- session_id: unique
- timestamp: ISO 8601
- messages_sent, messages_received
- bytes_in, bytes_out

## transport_http_requests.csv
- request_id: unique
- timestamp: ISO 8601
- method: HTTP verb
- status_code, bytes_in, bytes_out

## mcp_resource_access.csv
- resource_uri: e.g., db://leads
- timestamp: ISO 8601
- scope: read/write/search/consolidate
- operation: SELECT/INSERT/UPDATE/...
- success: bool
- actor: MCP-Server / Agent-Client / Optimizer-Worker

## segments.csv
- segment_id: unique
- name: segment name
- rules_json: JSON of inclusion rules
- description: text

## security_auth_events.csv
- event_id: unique
- timestamp: ISO 8601
- principal: service identity
- auth_mechanism: mTLS/OIDC/APIKey
- scope: authorized scope
- result: success/failure/expired/denied
- ip: source ip
