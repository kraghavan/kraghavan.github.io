---
layout: single
title: "I Built a Five-Agent SRE War Room and Grafana Watched Every Token"
date: 2026-04-25
categories: [llm-infrastructure, observability, grafana]
tags: [grafana, sigil-sdk, langgraph, anthropic, claude, multi-agent, sre, llm-ops]
excerpt: "Grafana AI Observability launched on April 21. I spent a few days building a five-agent incident response system to put it through its paces. Here's what actually happened."
header:
  og_image: /assets/images/sre-war-room/conversations-view.png
toc: true
toc_label: "Contents"
toc_icon: "fire"
---

On April 21, 2026, Grafana Labs quietly dropped a blog post: *AI Observability for Agents in Grafana Cloud*. I read it, got excited, and spent the next few days building something to put it through its paces. By April 25th I had five Claude agents handling fake production incidents — a P1 Redis split-brain, a DDoS on the CDN, a Postgres connection pool exhausted at 512/512 — and Grafana was watching every single token.

This is the story of that experiment. The plan, the setup, the instrumentation, the dashboard, and the parts that were genuinely painful. I'll be honest about all of it.

---

## The Context: Why This Matters

I've spent eleven years making distributed systems observable. Prometheus metrics, distributed traces, structured logs, SLOs, alerting pipelines. When a service degrades, I know where to look.

LLM agents have none of this. Until now, "observability" for an AI agent meant reading stdout and hoping the error messages were useful. Token costs were buried in billing dashboards three clicks deep. Quality was vibes-based — you knew something was off when your users told you. There was no equivalent of a trace showing "this is where the agent went wrong."

**Grafana AI Observability changes the mental model.** Conversations become traces. Token counts become metrics. Agent quality becomes an SLO. The existing Grafana workflows — alerting, dashboards, explore, drill-down — work on AI data the same way they work on infrastructure data.

That's the dream. This post is about testing it in the real world, four days after launch.

---

## The Plan: An AI-Powered SRE War Room

The idea was simple: build a multi-agent system that mirrors how a real SRE team handles incidents, then instrument it completely with Grafana's new Sigil SDK.

Five agents, each with a distinct role:

| Agent | SRE Equivalent | Model | What It Does |
|---|---|---|---|
| `triage-agent` | First responder | Claude Haiku | Reads the alert, classifies severity (P1-P4), identifies blast radius |
| `runbook-agent` | The senior who's seen it all | Claude Sonnet | Looks up the right runbook, produces an ordered remediation plan |
| `executor-agent` | The engineer who types commands | Claude Haiku | Simulates executing each runbook step (dry-run — no real infra harmed) |
| `postmortem-agent` | The reflective senior | Claude Sonnet | Writes a structured post-mortem: timeline, root cause, action items |
| `cost-cop-agent` | The FinOps person nobody invited | Claude Haiku | Reads Sigil telemetry, flags expensive agents, suggests model swaps |

The model selection was deliberate. Triage and execution are fast, structured tasks — Haiku handles them in under 3 seconds. Runbook planning and post-mortem writing need reasoning across complex context — that's Sonnet territory. The cost-cop agent using Haiku to audit the Sonnet agents is either very efficient engineering or deeply ironic. Probably both.

**Constraints:** Everything runs on an M4 Mac Mini. No GPU. No cloud Kubernetes. Total demo budget: under $2 in Anthropic API credits.

---

## The Setup: Twenty Fake Incidents That Feel Real

The system needed incident volume. I built a library of 20 canned PagerDuty-style alert payloads covering the greatest hits of production failure:

```json
{
  "incident_id": "INC-0001",
  "title": "payment-service: P1 OOMKilled pods - 3 replicas down",
  "severity_hint": "P1",
  "affected_services": ["payment-service", "payment-worker", "checkout-api"],
  "annotations": {
    "summary": "Pod OOMKilled in payments namespace",
    "description": "Container exceeded memory limit of 512Mi. 3/5 replicas down. Revenue impact estimated at $4,200/min."
  }
}
```

Each payload is realistic enough that the agents treat it as real. The triage agent doesn't know it's fake. It responds with genuine P1 urgency. That's the point — the mocking is deliberate and specific, not random noise.

The incident library covers: OOM kills, connection pool exhaustion, Redis split-brain, TLS certificate expiry, etcd slow reads, CDN DDoS, CrashLoopBackOff with Kafka lag, Elasticsearch cluster RED, Istio sidecar memory leaks, GPU OOM on inference workloads.

An "incident blaster" fires these scenarios at configurable intervals:

```bash
python -m scenarios.blaster --count 10 --interval 15
```

{% include figure
   image_path="/assets/images/sre-war-room/blaster-terminal.png"
   alt="The incident blaster terminal showing INC-0001 (DDoS) and INC-0002 (Redis split-brain) being processed"
   caption="The blaster in action. INC-0001 is a P1 CDN DDoS. INC-0002 is a Redis split-brain with dual master election. Both resolved in under 60 seconds each. The ✅ Sigil line confirms telemetry is shipping."
%}

---

## The Orchestration: LangGraph State Machine

Each incident runs through a LangGraph state machine. One incident = one conversation = one Grafana thread.

```
Incident Alert → triage-agent → runbook-agent → executor-agent → postmortem-agent
                                                                        ↓
                                              LangGraph IncidentState (shared)
```

The `conversation_id` is the key insight. Every agent that touches an incident gets the same `conversation_id`. When Grafana receives four Sigil generations with the same ID, it threads them into a single conversation. You see the complete incident lifecycle — from raw alert to finished post-mortem — in one view.

```python
with sigil_client.start_generation(
    GenerationStart(
        model=ModelRef(provider="anthropic", name="claude-haiku-4-5"),
        conversation_id=incident_id,   # same ID across all 4 agents
        agent_name="triage-agent",
        system_prompt=SYSTEM_PROMPT,
        tags={"incident_id": incident_id},
    )
) as rec:
    response = claude_client.messages.create(...)
    rec.set_result(
        input=[user_text_message(alert_text)],
        output=[assistant_text_message(response.content[0].text)],
    )
```

That's the entire instrumentation for one agent. Fifteen lines. The SDK handles token counting, latency tracking, and cost calculation automatically.

---

## The Dashboard: What Grafana Actually Shows

### AI Observability Landing Page

{% include figure
   image_path="/assets/images/sre-war-room/landing-page.png"
   alt="Grafana AI Observability landing page showing 24 conversations and 5 agents"
   caption="The AI Observability landing page after running 24 incidents. 5 agents registered, 24 conversations tracked. The tagline 'Actually useful AI O11y' is doing a lot of heavy lifting."
%}

The landing page summarises the last 24 hours: total conversations, average calls per conversation, token totals, cost. After running 10+ incidents you start to see the shape of your agent workload at a glance.

### Conversations View — The Main Event

{% include figure
   image_path="/assets/images/sre-war-room/conversations-view.png"
   alt="Grafana AI Observability Conversations view showing 24 conversations with agent breakdown"
   caption="24 conversations, 3.88 average calls per conversation (exactly right for a 4-agent pipeline). Each row is one incident. The activity chart shows the cadence of the blaster runs."
%}

Each row is one incident run. The columns show conversation ID, duration, call count, which agents participated, and which models were used. The conversation activity chart maps exactly to when the blaster was running — you can literally see when I paused it to debug something.

Clicking into a conversation is where it gets genuinely useful.

### Conversation Drilldown — Where the Real Value Lives

{% include figure
   image_path="/assets/images/sre-war-room/conversation-drilldown.png"
   alt="Single conversation drilldown showing triage-agent, runbook-agent, executor-agent, postmortem-agent with per-agent timing"
   caption="One incident, four agents. The timeline shows triage completes in 2.08s (Haiku, fast), runbook takes 8.57s (Sonnet, reasoning), executor at 7.58s, and postmortem at 21.61s (Sonnet, writing). Total: 37.8 seconds from alert to post-mortem."
%}

The per-agent breakdown on the left shows:
- `triage-agent`: **2.08s** (Haiku, exactly what we want)
- `runbook-agent`: **8.57s** (Sonnet reasoning through a complex runbook)
- `executor-agent`: **7.58s** (Haiku simulating kubectl commands)
- `postmortem-agent`: **21.61s** (Sonnet writing a full post-mortem)

The right panel shows the actual content — the triage agent's JSON classification, the runbook it matched, the alert context it processed. This is the debugging interface for AI agents that we've never had before.

{% include figure
   image_path="/assets/images/sre-war-room/conversation-postgres.png"
   alt="Conversation drilldown for Postgres P1 incident showing the full triage summary and matched runbook"
   caption="The Postgres P1 (connection pool exhausted — 512/512). The triage agent correctly identified P1 severity, blast radius single-region, and matched the DB Connection Pool Exhaustion runbook. All visible in Grafana."
%}

### Agents View — Who's Doing the Work

{% include figure
   image_path="/assets/images/sre-war-room/agents-view.png"
   alt="Grafana Agents view showing all 5 agents with generation counts and cost breakdown"
   caption="All five agents registered. The 'Top by Generations' chart shows near-equal usage across triage, runbook, executor, and postmortem (23 each) — exactly as expected. The cost-cop-agent ran twice (the two cost audit cycles)."
%}

The Agent Footprint section on the right is interesting: triage-agent is actually the most expensive by token volume despite using Haiku, because it runs on every incident. The postmortem-agent costs less in aggregate despite using Sonnet, because its output is dense and well-structured.

---

## The Cost Cop: An AI Auditing Other AIs

Every 10 incidents, the cost-cop agent wakes up, queries Grafana's API, and produces a FinOps report. That report — generated by a Haiku agent — gets shipped back to Grafana as its own conversation thread.

{% include figure
   image_path="/assets/images/sre-war-room/cost-cop-conversation.png"
   alt="Cost-cop agent conversation thread in Grafana showing FinOps analysis with model swap recommendations"
   caption="The cost-cop agent's conversation in Grafana. It identified that postmortem-agent was consuming 60% of total spend and recommended swapping to Haiku for an 83% cost reduction. This analysis is itself a Grafana conversation."
%}

The output is pointed:

> **postmortem-agent is your problem child.** 1-hour cost: $1.81 (60% of total spend). Projected monthly: $1,294. Postmortem generation is templated, deterministic work — **swap to Haiku immediately.** Projected savings: ~$1,050/month. Risk: Low. Test on 5% traffic first.

The cost-cop is Haiku auditing Sonnet. The meta-level here is not lost on me: a cheap model correctly identifying that expensive models are doing work they're overqualified for. This is the kind of FinOps signal that used to require a dedicated billing dashboard. Now it's a Grafana conversation.

---

## The Setup: What Was Hard

This is a four-day-old public preview. I am not going to pretend it was smooth. Here is the honest accounting:

### The Endpoint Discovery Problem

The Sigil SDK defaults to `localhost:8080`. The Grafana setup wizard shows `localhost:8080`. Neither of those is where you actually send data in Grafana Cloud.

After trying:
- `<my-grafana-username>.grafana.net/api/sigil/v1/generations:export` → 404
- `<my-grafana-username>.grafana.net/api/v1/generations:export` → 404
- `localhost:4317` (Alloy's OTLP port) → `UNIMPLEMENTED: unknown service sigil.v1.GenerationIngestService`
- `localhost:8080` → connection refused

The answer was buried in `Administration → Plugins → AI Observability → Connection tab`.

{% include figure
   image_path="/assets/images/sre-war-room/plugin-config.png"
   alt="AI Observability plugin configuration page showing sigil-prod-ca-east-0.grafana.net endpoint"
   caption="The actual endpoint: sigil-prod-ca-east-0.grafana.net. Instance ID: 1608023. Not documented anywhere I could find. Found by clicking through the plugin configuration UI."
%}

**The correct endpoint:** `https://sigil-prod-ca-east-0.grafana.net`  
**The correct auth:** Basic auth with Instance ID as username + Cloud Access Policy token as password  
**The correct token scope:** You need a Cloud Access Policy token with sigil write scope, *not* a service account token and *not* the OTLP token

Total time spent on authentication alone: approximately 90 minutes across five different token types, three endpoints, and two auth schemes.

For teams already running Grafana Cloud with established service accounts and Cloud Access Policies: this is a 10-minute task. For someone setting up from scratch on a 4-day-old SDK: it is a different experience.

### Running Alloy

Rather than using Docker Compose, we ran the Grafana Alloy container directly — simpler, more transparent, and easier to debug:

```bash
docker run -d \
  --name war-room-alloy \
  -p 4317:4317 -p 4318:4318 -p 8080:8080 -p 12345:12345 \
  -v $(pwd)/alloy/config.alloy:/etc/alloy/config.alloy:ro \
  --env-file .env \
  grafana/alloy:latest \
  run /etc/alloy/config.alloy --server.http.listen-addr=0.0.0.0:12345
```

{% include figure
   image_path="/assets/images/sre-war-room/alloy-logs.png"
   alt="Grafana Alloy Docker logs showing OTLP receivers starting on ports 4317 and 4318"
   caption="Alloy running correctly — GRPC receiver on 4317, HTTP receiver on 4318. The Prometheus remote write errors in the logs are Alloy's self-metrics hitting the wrong regional endpoint, which doesn't affect the agent telemetry path."
%}

### Claude Returns Markdown-Fenced JSON

Despite being told "output ONLY valid JSON — no markdown fences, no explanation," Claude Haiku wraps responses in ` ```json ``` ` blocks. Every time. Without fail.

```python
def strip_fences(text: str) -> str:
    """The fix that should have been in the system prompt."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()
```

One line of regex. Took longer to figure out than it should have. Now it's in every agent.

### The HTTP vs gRPC Transport Confusion

The sigil-sdk defaults to gRPC. Grafana Alloy doesn't implement the `sigil.v1.GenerationIngestService` gRPC protocol — it's an OTel collector, not a Sigil receiver. The fix was to use the HTTP exporter directly:

```python
from sigil_sdk.exporters.http import HTTPGenerationExporter

http_exporter = HTTPGenerationExporter(
    endpoint="https://sigil-prod-ca-east-0.grafana.net",
    headers={"Authorization": f"Basic {base64_creds}"},
)
config = ClientConfig(generation_exporter=http_exporter)
```

This isn't documented. I found it by reading the source code of the SDK package.

---

## What Experienced Teams Get vs. What New Users Face

**If your team already has Grafana Cloud** with established Cloud Access Policies, Alloy collectors running, and a platform team who knows where the endpoint configuration lives: this takes 30 minutes. Install `sigil-sdk`, add the `start_generation` context manager to your LLM calls, point at your existing sigil endpoint, done.

**If you're setting this up from scratch:** budget 3-4 hours for the first run. The SDK is solid. The dashboard is excellent. The undocumented corners are real but navigable. The endpoint discovery and auth story need better documentation before this can be called "zero-config."

The tagline on the landing page is "Actually useful AI O11y." I'd say: yes, once you're past setup.

---

## The Dashboard Experience: Seamless Once You're In

Here's the part that genuinely impressed me. Once the telemetry was flowing, the Grafana AI Observability UI is fast, well-designed, and immediately useful.

{% include figure
   image_path="/assets/images/sre-war-room/conversations-with-assistant.png"
   alt="Grafana Conversations view with AI Assistant panel showing analysis of 24 conversations"
   caption="The Grafana AI Assistant (built into the dashboard) automatically analysed the 24 conversations and flagged that token metrics showed 0 tokens recorded — a real instrumentation gap worth investigating. It also recommended enabling feedback collection for quality assessment."
%}

The AI Assistant panel on the right is a nice touch — it analyses your agent telemetry and surfaces actionable observations. It spotted that my token counts weren't flowing through (a known gap with how the Sigil HTTP exporter handles token reporting in this early version) and suggested next steps. A Grafana dashboard that uses AI to help you observe your AI agents. We've reached full recursion.

The tabs — Analytics, Agents, Conversations, Tools, Evaluation — cover every dimension of LLM observability:
- **Conversations**: the thread view, for debugging individual incidents
- **Agents**: the fleet view, for understanding workload distribution
- **Analytics**: the metrics view, for SLO and cost tracking
- **Evaluation**: the quality view, for LLM-as-judge scoring (Phase 4 for this project)

---

## The Numbers

After two runs of 10 incidents each:

| Metric | Value |
|---|---|
| Total conversations in Grafana | 24 |
| Average calls per conversation | 3.88 |
| Total agents registered | 5 |
| Average pipeline time | ~37 seconds |
| Incidents resolved | 11/20 |
| Incidents escalated | 3/20 |
| Incidents partial | 5/20 |
| Errors | 1 (malformed JSON, handled gracefully) |
| Estimated API cost for all runs | ~$1.50 |

The 1 error was a truncated JSON response from Claude — the post-mortem agent hit a complex scenario (CDN DDoS + multi-region blast radius) and the response exceeded the expected structure. The `strip_fences()` fix caught the fence issue; a similar safety net for truncated JSON is on the backlog.

---

## What's Next

### Phase 4: LLM-as-Judge Evaluation

The most compelling remaining capability is automated quality evaluation. The architecture is ready:

1. Configure a Grafana evaluator targeting `postmortem-agent` outputs
2. Rubric: "Does this post-mortem contain a specific root cause hypothesis?"
3. Alert: fire if quality score drops below 0.7
4. Test: intentionally degrade the postmortem system prompt and watch the alert fire

This closes the loop: instrument → observe → evaluate → alert. Agent quality becomes a production SLO. That's the end state.

### Extending to Real Projects

The patterns from this experiment map directly to production agent systems:
- **Any LangGraph pipeline**: add `start_generation` context managers, pass `conversation_id` through state
- **Existing Grafana Cloud setups**: five minutes to wire up once you have the endpoint
- **Cost tracking**: the cost-cop pattern works for any multi-agent system

---

## Conclusion

The SRE Incident War Room worked. Five agents handled 20 different incident types, Grafana tracked every conversation thread, and the cost-cop identified real optimisation opportunities — on an M4 Mac Mini, for under $2.

The setup has rough edges. The endpoint discovery is non-obvious. The auth story needs better documentation for first-time users.

None of that changes the core conclusion:

**Treating LLM agent conversations as first-class telemetry — alongside traces, metrics, and logs — is the right architecture for the agentic era.** The patterns established here: conversation threading, per-agent cost tracking, quality evaluation as an SLO — these are patterns that will matter at every company running AI agents in production.

As a staff SRE building toward LLM infrastructure roles, this experiment confirms the intersection I care about: SRE rigour applied to LLM systems. The tooling is here. The mental models transfer. The work is real.

---

## References

- [Grafana AI Observability Blog Post](https://grafana.com/blog/ai-observability-for-agents-in-grafana-cloud/)
- [Grafana AI Observability Docs](https://grafana.com/docs/grafana-cloud/machine-learning/ai-observability/)
- [Grafana Sigil SDK (GitHub)](https://github.com/grafana/sigil-sdk)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Anthropic Claude API](https://docs.anthropic.com)
- Project code: *sre-war-room* (GitHub — link coming)

---

*This post is part of my ongoing series on LLM inference infrastructure. Previous posts: [What Is LLM Inference, Really?](/llm-inference-deep-dive) and [Running LLM Inference on Apple Silicon](/apple-silicon-llm-inference).*