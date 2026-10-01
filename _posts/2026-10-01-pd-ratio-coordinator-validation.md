---
layout: single
title: "pd-ratio-coordinator — Does My Own Tool Actually Work?"
date: 2026-10-01
categories: [llm-infrastructure, inference]
tags: [llm-d, kubernetes, autoscaling, pd-disaggregation, vllm, prometheus, golang, gh200, a100, inference-architecture]
series: "LLM Inference from First Principles"
series_part: 8
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: >
  Part 6 mentioned a side project — a Kubernetes operator that rebalances
  prefill and decode replicas by watching real metrics instead of generic
  queue depth. It had never run against a real cluster. This is what
  happened when I finally rented a GPU and found out.
---

Part 6 of this series name-dropped [pd-ratio-coordinator](https://github.com/kraghavan/pd-ratio-coordinator)
in passing — a Kubernetes operator I'd written to autonomously rebalance
prefill and decode replica counts in an llm-d cluster, reacting to queue
*velocity* and a joint GPU budget instead of the generic queue-depth
checks a standard autoscaler uses. I cited it as "the part that connects
directly" to the memory-wall argument. What I didn't say at the time: it
had never actually run against a real cluster. ~1,600 lines of Go,
unit-tested, never GPU-tested.

This post is what happened when that stopped being true.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-verified.jpeg" alt="pd-ratio-coordinator going from ~1,600 lines, unit-tested, never run on a real cluster, to validated on real metrics, real load, real kubectl scale, in this session" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">From cited in passing to actually proven.</figcaption>
</figure>

---

## What pd-ratio-coordinator Actually Does

llm-d separates inference into two pools — prefill (compute-intensive
first-token generation) and decode (memory-bandwidth-intensive token
generation). WVA, the standard autoscaler, scales each pool independently
based on current queue depth. Two gaps that leaves:

**Spike blindness.** WVA reacts after a queue has already saturated.
pd-ratio-coordinator detects queue *velocity* — rate of growth — and
scales before saturation hits.

**Pool competition.** Without a joint constraint, prefill and decode can
each independently request more GPUs than the cluster has, leaving both
pools stuck Pending. pd-ratio-coordinator enforces
`prefill.replicas + decode.replicas <= gpuBudget` as a hard constraint,
always.

It also handles graceful decode drain (label a pod, wait for in-flight
sequences to finish, *then* scale down — decode pods hold live KV cache,
so killing one mid-sequence drops a user's request) and cooldown/hysteresis
to prevent oscillation.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-control-loop.jpeg" alt="Control loop: Prometheus scrapes vLLM /metrics every 5s, feeding AnalysePrefill (queue depth + velocity) and AnalyseDecode (TPOT p95 + KV cache %), into ComputeScaleDecision with a gpuBudget constraint, through CooldownGuard.Check, ending in kubectl scale prefill or label draining=true, poll, kubectl scale decode" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">Reconcile every 10 seconds.</figcaption>
</figure>

---

## Before Any of That Could Be Tested: Does It Even Compile?

No. `go build ./...` failed on the actual committed code (branch `KRAG-1`).
Two real bugs, not just missing dependencies:

1. **A DeepCopy type bug in `register.go`** — `PDRatioPolicyStatus`'s
   hand-written `DeepCopyInto` tried to append `[]interface{}` into a
   `[]metav1.Condition` field. Fixed to a proper typed copy.
2. **Unqualified constants in `scaler/decode.go`** — `BottleneckNone`,
   `BottleneckPrefill`, `BottleneckDecode`, `BottleneckBoth` were referenced
   as if they were local, but they're defined in `api/v1alpha1` and were
   never imported. Qualified all four.

`go.sum` had also never been committed, so the module wouldn't build for
anyone who cloned it fresh. And `config/crd/` — the directory the README
instructs users to `kubectl apply -f` — didn't exist. `controller-gen`
needed a `+groupName` marker that was never added (the group was only set
as a Go runtime variable, which static marker-scanning can't see).

All four fixed, pushed to `KRAG-1`, before any GPU was rented. If you're
going to validate a tool, "does it compile" is not optional pre-work.

---

## The Scope Decision: Skip Real P/D Disaggregation on Purpose

The repo's own `docs/testing.md` assumes real P/D disaggregation is
already running — NIXL over NVLink, prefill and decode on separate GPUs
with working KV transfer. Two problems with that assumption. First, [Part
5](/llm-infrastructure/inference/2026/04/21/llm-d-pd-disaggregation.html)
already found that a single time-sliced node has no RDMA path — NIXL
fails, the architecture collapses into aggregated serving. Second, the
"1x GH200 + H100" combo instance the docs assume may not even be a real
Lambda SKU — the console only ever showed separate GH200, H100, A10, and
A100 listings.

Here's the way out: **pd-ratio-coordinator doesn't actually care whether
KV transfer between prefill and decode is real.** Its entire job is
reading Prometheus metrics and calling the Kubernetes API to scale
Deployments. It never inspects NIXL, never touches KV transfer
correctness. So the tool's whole value proposition — velocity detection,
joint GPU budget, graceful drain, cooldown — can be validated with two
plain vLLM Deployments labeled `prefill` and `decode`, real Locust load,
real Prometheus scraping, real `kubectl scale` calls, and zero dependency
on cross-GPU KV transfer working at all.

That means **one GPU is enough.** No NVLink, no combo SKU, no repeating
Part 5's unresolved problem for a reason that has nothing to do with what
this tool actually does.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-scope-triage.jpeg" alt="Scope triage: pd-ratio-coordinator's job is to read Prometheus and call the k8s API. NIXL / cross-GPU KV transfer correctness is not this tool's job (Part 5's unresolved problem, skip it); real metrics + real load + real kubectl scale is this tool's job, testable on one GPU" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">One GPU is enough, here's why.</figcaption>
</figure>

---

## Setup

Single-node k3s on a rented A100 (40GB). Two plain vLLM Deployments,
Qwen3-0.6B, labeled `prefill` and `decode`, both sharing the one physical
GPU via `--gpu-memory-utilization=0.2` each rather than requesting
isolated `nvidia.com/gpu` units — deliberately, so multiple replicas could
coexist on one card. Lightweight annotation-based Prometheus (no Helm
chart). The controller itself run via `make run` directly on the host,
not as an in-cluster pod — simpler to iterate on, closer to how I was
actually debugging it.

### What broke getting the cluster up (kept in full, same as every other post in this series)

1. **No GPU access at all, at first.** Skipping `nvidia.com/gpu` resource
   requests also meant containerd never injected GPU device access into
   the containers — `RuntimeError: Failed to infer device type`. Fixed
   with a Kubernetes `RuntimeClass` object (`handler: nvidia`) and
   `runtimeClassName: nvidia` on the pod specs, rather than forcing nvidia
   as the cluster-wide containerd default (which would also apply to
   Prometheus, which needs no GPU at all).
2. **`containerd: failed to unmarshal TOML: toml: table containerd already
   exists`**, then `table nvidia already exists` on the next attempt. Turns
   out k3s on this GPU-ready box already auto-detects
   `nvidia-container-runtime` and configures a working `nvidia` runtime
   handler on its own. My custom containerd template wasn't needed at all
   — removed entirely once confirmed.
3. **Stale containerd-shim processes blocked a clean k3s restart.** Same
   "child process survives killing the parent" shape as the zombie
   EngineCore issue from the earlier vLLM post — containerd shims are
   *designed* to survive their parent's death, which is exactly what makes
   them annoying here. Fixed with k3s's own `k3s-killall.sh` before
   restarting.
4. **In-cluster service DNS doesn't resolve from the host.** Running the
   controller outside the cluster meant `http://prometheus.pd-validation:9090`
   couldn't resolve. Fixed with a persistent port-forward and pointing
   `prometheusURL` at `localhost`.
5. **`go run` spawns a child binary that survives killing the parent.**
   The zombie-child pattern a third time, now in Go's own tooling —
   `pkill -f 'go run main.go'` doesn't touch the actual compiled binary at
   `/tmp/go-build.../exe/main`. Had to find and kill it by PID directly.
6. **Two real metric-name drifts in vLLM v0.30.0**, found by diffing the
   operator's hardcoded PromQL against the server's actual `/metrics`
   output:
   - `vllm:gpu_cache_usage_perc` → renamed to `vllm:kv_cache_usage_perc` —
     the *same* rename [Part 2](/llm-infrastructure/inference/2026/04/16/vllm-ollama-apple-silicon-experiment2.html)
     already documented as a gotcha, on a different vLLM version. This
     metric name has been drifting for a while, not a one-off.
   - `vllm:time_per_output_token_seconds` → renamed to
     `vllm:request_time_per_output_token_seconds`.

   Both fixed in `internal/metrics/prometheus.go`, rebuilt, redeployed.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-what-broke.jpeg" alt="Six things that broke: no GPU access, fixed with RuntimeClass; TOML table conflict, k3s already had it built in; containerd-shim zombie, same pattern 3rd time; in-cluster DNS unreachable from host; go run spawns a zombie binary too; two vLLM metric names drifted" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">Six real things, kept in.</figcaption>
</figure>

---

## Results

Four tests, adapted from the repo's own `docs/testing.md`, de-risked as
described above.

### Test 1 — TPOT SLO breach → scale decode: passed

30 concurrent users, 60s, against decode. **1,970 requests, 0 failures**,
p50 530ms, p95 560ms.

Real finding before the real pass: measured `decode.tpot_p95_ms` sat at
**9.5ms** under this load — far below the first "aggressive-sounding" 20ms
SLO I set, which never triggered anything. Qwen3-0.6B decodes faster than
that guess, even sharing a GPU at 20% utilization. Had to lower the SLO to
5ms — below the measured baseline — to get a clean, deterministic breach.
Stated plainly: that's a hand-tuned threshold to exercise the code path,
not a realistic production number for this model.

Once it triggered: `tpot_slo_breach(10ms>5ms)` in the log, cooldown
respected, then a real scale action — `"scaled decode", "from": 1, "to": 2`
— and a real second decode pod came up.

### Test 2 — queue velocity spike → scale prefill: inconclusive, and that's the honest result

Escalated three times — 20 users, then 150, then 300. Final run:
**12,517 requests, 0 failures**, 520.7 req/s sustained, p50 470ms, p95
560ms, p99 680ms.

`prefill.queue` stayed at **zero for the entire duration of all three
runs** — confirmed both by the controller's own 10-second reconcile
snapshots and by polling Prometheus directly every 2 seconds during the
final burst. Even at 300 concurrent requests and 500+ req/s, nothing ever
backed up into a waiting state.

Most likely reason: vLLM's continuous batching on a 0.6B model admits
requests into the running batch fast enough that nothing queues, at any
concurrency this setup could throw at it. That's a property of this
model/hardware pairing, not a bug in the detection code — `AnalysePrefill`
already has passing unit tests against synthetic queue data. What this
round couldn't do is produce a real-world backlog to exercise that path
operationally. A v2 validation would need either a much larger model or a
deliberately starved `--max-num-seqs` to actually generate queue depth.

### Test 3 — GPU budget enforcement: passed

Budget set equal to the current total (prefill=1, decode=2, budget=3). 30
users, 40s, against decode. **1,376 requests, 0 failures**, p50 530ms,
p95 550ms.

Real pressure was present — `decode.under_pressure: true`,
`tpot_slo_breach(10ms>5ms)` — and decode replicas stayed at 2 the entire
time. The controller correctly refused to scale past budget even with
genuine, sustained pressure asking it to. `prefill + decode <= gpuBudget`
held throughout. Clean, unambiguous result.

### Test 4 — drain before scale-down: nuanced, and worth explaining rather than rounding up

8 users, 70s, against decode. **696 requests, 0 failures**, p50 470ms,
p95 480ms. Budget cut from 3 to 2 mid-flight to force a scale-down.

The `llmd.io/draining=true` label was applied correctly to the targeted
pod. But the drain **timed out after its configured 20 seconds** —
`"drain timeout, force scaling down"` — and the controller force-scaled
down anyway.

The more important finding is underneath that: this setup uses a plain
Kubernetes `Service` (kube-proxy round-robin), not llm-d's actual EPP
gateway. The draining label is designed to be read by that EPP to stop
routing new requests to the pod — a plain Service has no concept of the
label at all. So the drain mechanism's *write path* — apply label, poll
in-flight request count, force-scale on timeout — is confirmed to execute
correctly end to end. Its *actual protective effect* was not validated
here, because half the mechanism (routing exclusion) isn't present in
this de-risked setup. Zero failures is good news, but it may simply mean
nothing happened to be in-flight on that specific pod at the moment it
was killed, not that the drain protected it. Real validation of this
guarantee needs llm-d's EPP in front of the pods — explicitly outside
this round's scope, not a result I'm claiming.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-results.jpeg" alt="Results: Test 1 TPOT breach to scale decode, passed, 1,970 requests 0 failures. Test 2 velocity spike to scale prefill, inconclusive, 12,517 requests 0 failures, queue never left 0. Test 3 GPU budget enforcement, passed, 1,376 requests 0 failures. Test 4 drain before scale-down, nuanced, 696 requests 0 failures, routing exclusion unverified" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">Four real answers, not four passes.</figcaption>
</figure>

### Summary

| Test | Result | Requests | Failures | p50 | p95 |
|---|---|---|---|---|---|
| TPOT breach → scale decode | **Passed** | 1,970 | 0 | 530ms | 560ms |
| Velocity spike → scale prefill | **Inconclusive** (real limitation) | 12,517 | 0 | 470ms | 560ms |
| GPU budget enforcement | **Passed** | 1,376 | 0 | 530ms | 550ms |
| Drain before scale-down | **Nuanced** (write path confirmed, protection unverified) | 696 | 0 | 470ms | 480ms |

**16,559 total requests across every test. Zero failures, throughout.**

---

## What This Means

Two of four mechanisms are cleanly proven: the velocity/SLO detection
logic reacts to real signals and makes real scaling decisions, and the
joint GPU budget constraint holds even under genuine competing pressure.
One is a real, named limitation — this model/hardware pairing couldn't
produce the queue backlog needed to exercise spike detection, through no
fault of the detection code itself. One is a genuinely nuanced result
that I'd rather report honestly than round up to a pass: the drain
mechanism's bookkeeping works, but proving it protects real traffic needs
a piece of infrastructure (llm-d's EPP) this round deliberately scoped
out.

That's four real answers, not four passes. I think that's more useful to
anyone deciding whether to actually run this operator than a clean
scorecard would have been.

**Open for v2:** a workload that can genuinely saturate prefill admission
(bigger model, or a resource-starved config), and a real llm-d EPP in
front of the pods to validate the drain label's actual routing-exclusion
effect, not just its bookkeeping.

---

## The Scripts

Everything here — the k8s manifests, the Locust load generator, the setup
and test-runner scripts, the Go fixes — is in
[gpu-labs](https://github.com/kraghavan/gpu-labs/tree/main/pd-ratio-coordinator-validation),
alongside the fixed-up [pd-ratio-coordinator](https://github.com/kraghavan/pd-ratio-coordinator/tree/KRAG-1)
repo itself. If you find a way to get real queue backlog out of a small
model, or want to wire up a real EPP for the drain test, I'd genuinely
like to know.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-infrastructure/pd-ratio-coordinator-whats-next.jpeg" alt="What's next: a bigger model or starved max-num-seqs to get real queue backlog, and a real llm-d EPP in front to validate drain's routing exclusion. Scripts at github.com/kraghavan/gpu-labs" style="width:100%;">
  <figcaption style="font-size:0.9rem;color:#888;">16,559 requests, zero failures, two open questions.</figcaption>
</figure>

---

*Experiments run on Lambda Cloud, 1x A100 (40GB SXM4), k3s, vLLM v0.30.0,
Qwen3-0.6B, pd-ratio-coordinator branch KRAG-1. Platform engineer with
11+ years in distributed systems going deep on LLM serving
infrastructure.*

*[GitHub](https://github.com/kraghavan) · [LinkedIn](https://linkedin.com/in/karthikaraghavan)*
