---
layout: single
title: "Projects"
permalink: /projects/
author_profile: true
---

# Open Source Projects

Tools I've built during my career transition — each one solving a real problem I encountered.

---

## 🛡️ inference-sentinel

**Privacy-aware LLM routing gateway**

Routes LLM prompts to local or cloud inference based on data sensitivity. Detects PII in real-time and enforces routing policies with session-level lockdown.

| Metric | Value |
|--------|-------|
| Classification accuracy | 97.5% |
| Classification latency | 0.16ms |
| Cost savings | 68% vs all-cloud |

**Key features:**
- Hybrid classifier (Regex + NER) with 4-tier privacy taxonomy
- One-way trapdoor session stickiness — once PII is detected, session is locked to local
- Shadow mode for local vs cloud quality comparison
- Closed-loop controller with drift detection
- Full observability: Prometheus, Grafana, Loki, Tempo

**Tech:** Python, FastAPI, Ollama, Docker, Prometheus, OpenTelemetry

[GitHub](https://github.com/kraghavan/inference-sentinel){: .btn .btn--primary} [Blog Post](/llm/infrastructure/smart%20gateway/python/2026/03/20/inference-sentinel-architecture.html){: .btn .btn--info}

---

## ⚡ go-fast-token

**Parallel BPE tokenizer in Go**

High-performance tokenizer achieving 3.04M tokens/sec on Apple M4. Implements auto-tuning worker selection for optimal parallelization.

| Metric | Value |
|--------|-------|
| Throughput | 3.04M tokens/sec |
| Improvement | 45% via auto-tuning |
| Encoding | o200k_base (GPT-4o compatible) |

**Key features:**
- Parallel chunk processing with optimal worker selection
- Fixed chunker to preserve BPE merge boundaries
- LLM verification test suite
- Published to pkg.go.dev

**Tech:** Go, BPE, Concurrency

[GitHub](https://github.com/kraghavan/go-fast-token){: .btn .btn--primary} [pkg.go.dev](https://pkg.go.dev/github.com/kraghavan/go-fast-token){: .btn .btn--info}

---

## 🗃️ schema-travels

**SQL-to-NoSQL migration analyzer**

Analyzes SQL schemas and recommends NoSQL data models. Detects access patterns, suggests partition keys, and generates migration scripts.

**Key features:**
- Automatic access pattern detection
- DynamoDB single-table design recommendations
- Confidence scoring for migrations
- Terraform HCL and NoSQL Workbench JSON output

**Tech:** Python, SQL parsing, DynamoDB

[GitHub](https://github.com/kraghavan/schema-travels){: .btn .btn--primary} [PyPI](https://pypi.org/project/schema-travels/){: .btn .btn--info}

---

## 🤖 model-council

**Multi-model AI code review**

Aggregates code review feedback from multiple LLMs (Claude, GPT-4, Gemini) with semantic deduplication to surface unique insights.

**Key features:**
- Deep analysis mode with context caching
- Cross-PR issue fingerprinting (NEW/UNRESOLVED/RECURRING)
- Embeddings-based semantic matching (cosine similarity 0.85 threshold)
- Schema versioning for upgrade paths

**Tech:** Python, OpenAI API, Anthropic API, Embeddings

[GitHub](https://github.com/kraghavan/model-council){: .btn .btn--primary}

---

## 📄 indicflow

**Multilingual Indian document analysis**

CLI tool for analyzing documents in Indian languages using Gemini 2.0 Flash. Supports Hindi, Tamil, Telugu, Bengali, Urdu, and more.

**Key features:**
- Pipeline mode: batch translate + keyword scan
- Deep mode: word frequency, sentiment, bias detection
- PDF/DOCX/PPTX/image support via Vision API
- Urdu RTL text reshaping

**Tech:** Python, Gemini API, PDF processing

[GitHub](https://github.com/kraghavan/indicflow){: .btn .btn--primary} [PyPI](https://pypi.org/project/indicflow/){: .btn .btn--info}

---

## 🔧 agent-forge

**Multi-agent system generator**

Framework for building multi-agent AI systems with human-in-the-loop escalation.

**Key features:**
- Advisory HITL (capacity escalation)
- Collaborative HITL (AI disagreement tiebreaker)
- Chaos injector for resilience testing
- Streamlit dashboard for monitoring

**Tech:** Python, FastAPI, PostgreSQL, Streamlit

[GitHub](https://github.com/kraghavan/agent-forge){: .btn .btn--primary}

---

## Want to Collaborate?

I'm always interested in contributing to infrastructure and ML tooling projects. If you're building something interesting, [let's connect](https://linkedin.com/in/karthikaraghavan).
