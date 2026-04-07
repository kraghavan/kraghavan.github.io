---
layout: single
title: "Projects"
permalink: /projects/
author_profile: true
---

<style>
.project-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

@media (max-width: 768px) {
  .project-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .project-grid {
    grid-template-columns: 1fr;
  }
}

.project-tile {
  background: var(--background-color, #fff);
  border: 1px solid #e1e1e1;
  border-radius: 12px;
  padding: 1.25rem;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  color: inherit;
  display: block;
}

.project-tile:hover {
  border-color: #4dabf7;
  transform: translateY(-3px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.project-tile .icon {
  font-size: 28px;
  margin-bottom: 12px;
}

.project-tile h3 {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #4dabf7;
}

.project-tile .hook {
  font-size: 14px;
  color: #666;
  line-height: 1.5;
  margin: 0;
}

.project-detail {
  background: #2a2a2e !important;
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 8px;
  display: none;
  border: 1px solid #3a3a3e !important;
  color: #e0e0e0 !important;
}

.project-tile:hover + .project-detail,
.project-detail:hover {
  display: block;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.detail-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 4px 0;
  color: #e8e8e8 !important;
}

.detail-hook {
  font-size: 14px;
  color: #4dabf7 !important;
  font-style: italic;
  margin: 0;
}

.detail-tech {
  font-size: 12px;
  color: #999 !important;
}

.detail-desc {
  font-size: 15px;
  line-height: 1.6;
  margin: 0 0 16px 0;
  color: #c8c8c8 !important;
}

.detail-footer {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}

.detail-stats {
  color: #999 !important;
}

.detail-links a {
  color: #4dabf7 !important;
  text-decoration: none;
  margin-left: 12px;
}

.detail-links a:hover {
  text-decoration: underline;
  color: #74c0fc !important;
}
</style>

<div class="project-grid">

  <div class="project-tile" onclick="showDetail('sentinel')">
    <div class="icon">🛡️</div>
    <h3>inference-sentinel</h3>
    <p class="hook">What if your LLM gateway could forget how to trust the cloud?</p>
  </div>

  <div class="project-tile" onclick="showDetail('token')">
    <div class="icon">⚡</div>
    <h3>go-fast-token</h3>
    <p class="hook">3M tokens/sec because Python wasn't angry enough</p>
  </div>

  <div class="project-tile" onclick="showDetail('schema')">
    <div class="icon">🗃️</div>
    <h3>schema-travels</h3>
    <p class="hook">Your SQL schema wants to be a MongoDb or DynamoDB table. I translate.</p>
  </div>

  <div class="project-tile" onclick="showDetail('council')">
    <div class="icon">🤖</div>
    <h3>model-council</h3>
    <p class="hook">What if 3 AIs reviewed your PR and actually agreed?</p>
  </div>

  <div class="project-tile" onclick="showDetail('indic')">
    <div class="icon">📄</div>
    <h3>indicflow</h3>
    <p class="hook">NLP for the other 1.4 billion people</p>
  </div>

  <div class="project-tile" onclick="showDetail('forge')">
    <div class="icon">🔧</div>
    <h3>agent-forge</h3>
    <p class="hook">Multi-agent chaos, with a human escape hatch</p>
  </div>

</div>

<div id="detail-panel" class="project-detail" style="display: block; background: #2a2a2e; border: 1px solid #3a3a3e;">
  <p style="color: #888; margin: 0;">👆 Click a project to see details</p>
</div>

<script>
const projects = {
  sentinel: {
    title: "inference-sentinel",
    hook: "What if your LLM gateway could forget how to trust the cloud?",
    desc: "Privacy-aware LLM routing with a one-way trapdoor. Once a session sees PII, it's locked to local inference forever. No takebacks. Features hybrid classification (regex + NER), shadow mode for quality comparison, and a closed-loop controller that knows when to stay quiet.",
    tech: "Python, FastAPI, Ollama, Prometheus, Grafana",
    stats: "97.5% accuracy • 0.16ms classification • 68% cost savings",
    github: "https://github.com/kraghavan/inference-sentinel",
    blog: "/llm/infrastructure/smart%20gateway/python/2026/03/20/inference-sentinel-architecture.html",
    links: '<a href="https://github.com/kraghavan/inference-sentinel">GitHub</a><a href="/llm/infrastructure/smart%20gateway/python/2026/03/20/inference-sentinel-architecture.html">Blog Post</a>'
  },
  token: {
    title: "go-fast-token",
    hook: "3M tokens/sec because Python wasn't angry enough",
    desc: "Parallel BPE tokenizer that auto-tunes worker count based on input size. Chunks intelligently to preserve merge boundaries. Sometimes the boring optimization is the right one.",
    tech: "Go, BPE, Concurrency",
    stats: "3.04M tokens/sec on M4 • 45% improvement via auto-tuning • o200k_base support",
    github: "https://github.com/kraghavan/go-fast-token",
    links: '<a href="https://github.com/kraghavan/go-fast-token">GitHub</a><a href="https://pkg.go.dev/github.com/kraghavan/go-fast-token">pkg.go.dev</a>'
  },
  schema: {
    title: "schema-travels",
    hook: "Your SQL schema wants to be a DynamoDB table. I translate.",
    desc: "Analyzes SQL schemas, detects access patterns, suggests partition keys, and generates NoSQL based schema. Outputs Terraform HCL and NoSQL Workbench JSON. Because migration planning shouldn't require a whiteboard.",
    tech: "Python, SQL parsing, DynamoDB, MongoDB, Terraform",
    stats: "On PyPI • v2.3.0 • NoSQL database design recommendations",
    github: "https://github.com/kraghavan/schema-travels",
    links: '<a href="https://github.com/kraghavan/schema-travels">GitHub</a><a href="https://pypi.org/project/schema-travels/">PyPI</a>'
  },
  council: {
    title: "model-council",
    hook: "What if 3 AIs reviewed your PR and actually agreed?",
    desc: "Multi-model code review with semantic deduplication. Claude, GPT, and Gemini walk into a repo... and their feedback gets deduplicated via embeddings so you see unique insights, not three versions of 'add error handling'.",
    tech: "Python, OpenAI, Anthropic, Embeddings",
    stats: "Cosine similarity 0.85 threshold • Cross-PR fingerprinting • NEW/RECURRING labels",
    github: "https://github.com/kraghavan/model-council",
    links: '<a href="https://github.com/kraghavan/model-council">GitHub</a>'
  },
  indic: {
    title: "indicflow",
    hook: "NLP for the other 1.4 billion people",
    desc: "Sentiment & context analysis for Hindi, Tamil, Telugu, Bengali, Urdu, and more documents. Supports PDF, DOCX, images via Vision API. Handles Urdu RTL reshaping because text direction shouldn't be an afterthought.",
    tech: "Python, Gemini API, Vision, PDF processing",
    stats: "8 languages • Pipeline + Deep analysis modes • RTL support",
    github: "https://github.com/kraghavan/indicflow",
    links: '<a href="https://github.com/kraghavan/indicflow">GitHub</a><a href="https://pypi.org/project/indicflow/">PyPI</a>'
  },
  forge: {
    title: "agent-forge",
    hook: "Multi-agent chaos, with a human escape hatch",
    desc: "Framework for building multi-agent systems with human-in-the-loop escalation. Two modes: Advisory (capacity overflow) and Collaborative (AI disagreement tiebreaker). Because sometimes the AI should ask for help.",
    tech: "Python, FastAPI, PostgreSQL, Streamlit",
    stats: "Advisory + Collaborative HITL • Chaos injector • Audit logging",
    github: "https://github.com/kraghavan/agent-forge",
    links: '<a href="https://github.com/kraghavan/agent-forge">GitHub</a>'
  }
};

function showDetail(key) {
  const p = projects[key];
  const panel = document.getElementById('detail-panel');
  panel.innerHTML = `
    <div class="detail-header">
      <div>
        <h3 class="detail-title">${p.title}</h3>
        <p class="detail-hook">${p.hook}</p>
      </div>
      <span class="detail-tech">${p.tech}</span>
    </div>
    <p class="detail-desc">${p.desc}</p>
    <div class="detail-footer">
      <span class="detail-stats">${p.stats}</span>
      <span class="detail-links">${p.links}</span>
    </div>
  `;
  panel.style.display = 'block';
  panel.style.background = '#2a2a2e';
  panel.style.border = '1px solid #3a3a3e';
  panel.style.color = '#c8c8c8';
  
  // Highlight selected tile
  document.querySelectorAll('.project-tile').forEach(t => t.style.borderColor = '#e1e1e1');
  event.currentTarget.style.borderColor = '#4dabf7';
}
</script>

---

## Want to collaborate?

I'm always interested in contributing to infrastructure and ML tooling projects. If you're building something interesting, [let's connect](https://linkedin.com/in/karthikaraghavan).
