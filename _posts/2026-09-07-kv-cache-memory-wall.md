---
layout: single
title: "When Prefill Stops Being Compute-Bound — The KV Cache Memory Wall and What It Does to Your Scheduler"
date: 2026-09-07
categories: [llm-infrastructure, inference]
tags: [kv-cache, paged-attention, gqa, mla, fp8, turboquant, quantization, mamba, ssm, linear-attention, hbm4, rubin, pd-disaggregation, lmcache, vllm, inference-architecture]
series: "LLM Inference from First Principles"
series_part: 6
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: >
  Parts 2 through 5 were experiments — I ran the thing and reported what the
  dashboards said. This one is a landscape post, so I have flagged every number
  by whether it is measured by me, published by someone else, or my own
  extrapolation. The through-line: once enough of your prefill is loading cached
  KV rather than computing it, prefill stops being compute-bound, and most of the
  optimisations we built on top of that assumption stop working.
---

Every post in this series so far has been an experiment. I stood something up, pointed Locust at it, and reported what Grafana showed — including [the run where NIXL showed "No Data" on every KV transfer metric](/llm-infrastructure/inference/2026/04/21/llm-d-pd-disaggregation.html) and the architecture collapsed back into aggregated serving.

This post is a landscape post, and I want to be explicit about that because the rest of the series has trained you to expect numbers I generated myself.

- **Measured by me:** only the results already published in parts 4 and 5 — an 81.3% prefix cache hit rate, and the aggregated-vs-broken-P/D comparison of 15ms vs 54.9ms TTFT and 260ms vs 1,690ms E2E p50. Small model, single GPU. Reused as an anchor, not presented as new work.
- **Published by others:** everything else, linked, with the provenance stated where a source is a preprint, a vendor blog, or a single secondary report.
- **Mine:** the predictions in the second-to-last section, labelled with confidence and reasoning so you can argue with the reasoning rather than the conclusion.

The through-line is not "the KV cache is big." It is narrower and more useful:

> **Once enough of your prefill is loading cached KV rather than computing it, prefill stops being compute-bound — and P/D disaggregation, token-budget scheduling, and hardware specialisation were all built on the assumption that it is.**

That claim is not mine. It comes from Meng, Lee and Wang's [*Understanding Bottlenecks for Efficiently Serving LLM Inference with KV Offloading*](https://arxiv.org/abs/2601.19910) (arXiv 2601.19910v1, 16 Dec 2025 — a preprint in MLSys submission format, not peer-reviewed at time of writing). I read it in full for this post, and it reframed the whole thing. Most of the analytical spine below is theirs; I have tried to be scrupulous about which parts.

The organising split I use in the middle section — Transformer-preserving versus Transformer-replacing — is [Devansh's framing](https://machine-learning-made-simple.medium.com/transformers-vs-mamba-vs-linear-attention-who-wins-long-context-f1dc8ceb5ede), not mine. Credit where it's due; it's the clearest way I've seen to organise a messy field.

---

## Terms, Defined Once

- **Prefill** — processing the input prompt. Traditionally compute-bound, drawing 70–100% of peak GPU power.
- **Decode** — generating tokens one at a time. Memory-bandwidth-bound, 20–40% of peak power.
- **B_kv** — bytes of KV cache per token. Set by layers, heads, head dimension, and precision.
- **κ_ratio** — cached tokens divided by newly-prefilled tokens, for a request. A workload property.
- **κ_crit** — the ratio at which prefill flips from compute-bound to memory-bound. A property of the model and the hardware.
- **TTFT / ITL** — the metrics I've been reporting since part 2.

κ_ratio and κ_crit are Meng et al.'s formulation. The rest of this post leans on them heavily, because once you have those two numbers the rest of the field organises itself.

---

## Why the Cache Became the Bill

vLLM's own framing is the honest version of the hook I originally wanted to write: for standard full-attention decoders, [the KV cache often dominates GPU memory at 128k+ contexts, and each decode step must read a large fraction of it](https://vllm.ai/blog/2026-04-22-fp8-kvcache).

*(I originally had precise-looking figures here — "70–90% of VRAM at 1M tokens." I couldn't trace them to a primary source, so they're gone. Qualitative and sourced beats precise and unattributable.)*

Two second-order effects matter more than the raw memory number:

**Batch size is memory-gated.** Decode is bandwidth-bound, so throughput comes from batching. Every byte of cache you don't store is a byte spent on more concurrent sequences.

**The cache has to be read every decode step.** Capacity is one constraint; bandwidth is the other. This is why the hardware section later matters in two dimensions.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/kappa-ratio-vs-kappa-crit.png" alt="Log-scale chart showing compute-bound and memory-bound regions, with measured kappa_crit thresholds far to the left of real workload kappa_ratio medians" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
    <strong>Figure 1 — The gap that defines the problem.</strong>
    Measured κ_crit values land at 1–2 on H100;
    real workloads sit at 100 (multi-turn chat) to 10,000 (financial document Q&amp;A).
    Even NVLink C2C, at 14× PCIe bandwidth, doesn't close it for document workloads.
    Figures from Meng, Lee &amp; Wang (arXiv 2601.19910), a preprint.
  </figcaption>
</figure>

---

## Six Phases

Each phase solved the previous one's bottleneck and created the next.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/kv-cache-six-phase-evolution.png" alt="Timeline of six KV cache optimisation phases, each labelled with the bottleneck it created" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
    <strong>Figure 2 — Six phases, 2020–2026.</strong>
    Each solves the previous bottleneck and creates the next.
    The levers are largely orthogonal, which is why production stacks stack several.
  </figcaption>
</figure>

### Phase 1 — Naive caching (2020–2022)

Contiguous buffers sized for max sequence length. Correct, simple, wasteful.

**Created:** a memory-management problem.

### Phase 2 — Paged memory (2023)

[PagedAttention](https://arxiv.org/abs/2309.06180) borrowed OS virtual memory: fixed-size blocks allocated on demand, a block table mapping logical to physical, copy-on-write for forked sequences. It's now the substrate under vLLM, SGLang, TGI and TensorRT-LLM. Every experiment in parts 3–5 ran on it without my thinking about it once.

**Created:** the cache is efficiently stored but still fully materialised per request.

### Phase 3 — Head sharing: MQA and GQA (2023–2024)

MQA has all query heads share one K/V head. GQA groups them, giving `n_h/G` savings and a tunable dial between MHA and MQA. GQA is the default in Llama 3, Mistral and Qwen.

**The constraint that matters:** this is a training-time decision. You inherit it; you can't retrofit it. That theme repeats — the cheapest wins are the ones you don't control.

**Created:** shared prefixes across requests are still recomputed.

### Phase 4 — Reuse: prefix caching and radix trees (2024)

The phase I actually measured, in [part 4](/llm-infrastructure/inference/2026/04/19/llm-d-epp-prefix-cache-results.html).

Token repetition is pervasive — multi-turn conversations accumulate context, document Q&A runs many queries over the same document, code tools re-analyse the same project. Prefix caching finds the longest common prefix and computes KV only for the suffix, via hash matching (vLLM) or radix trees (SGLang).

My Locust workload used a 4:1 tenant-session to cold-request ratio with long system prompts, and the EPP held an **81.3% prefix cache hit rate**. Four in five requests skipped most of their prefill.

**Created:** and this is the pivot of the whole post — *prefix caching is what makes prefill memory-bound.* The more successfully you reuse, the smaller T gets, the larger κ_ratio gets, and the further you slide into the regime where you're not computing anything, just moving bytes. Phase 4 is a victim of its own success.

### Phase 5 — Low-rank compression: MLA (2024–2026)

DeepSeek's Multi-head Latent Attention stores a low-rank latent projection instead of full keys and values.

The measured numbers, from Meng et al.'s Table 3: **DeepSeek-V3 at 70 KB/token against 192 KB for Qwen3-235B-A22B and 328 KB for Llama-3.1-70B — a 2.7–4.7× reduction.** For scale, Llama-3.1-405B sits at 516 KB/token, making DeepSeek-V3's cache 14% the size.

That translates directly: DeepSeek-V3's κ_crit is 4.6× higher than Qwen3-235B's, extending the compute-bound region to κ_ratio ≈ 40 versus ≈ 8.

**Two caveats, and I've corrected an earlier overstatement here.** In a draft of this post I wrote that MLA's benefits couldn't be demonstrated because serving implementations weren't mature. That's broader than the source supports. What the paper actually says is narrower: they attempted to evaluate **DeepSeek-V2** and hit implementation-specific overheads that prevented clean isolation of PCIe transfer time. That's a measurement-harness problem in their offloading setup, not a general verdict on MLA in production. The honest takeaway is smaller but still real: benchmark MLA on your own stack rather than assuming paper numbers transfer.

The second caveat is the paper's own conclusion, and it's blunt: MLA delays the memory-bound transition but does not eliminate it. Both DeepSeek-V3 and Qwen3-235B become bandwidth-limited at κ_ratio between 100 and 1,000 — which is where real workloads live.

### Phase 6 — Quantization (2025–2026)

The one fully post-hoc lever. And this is where I had to throw out my first draft entirely.

I originally published a tidy table giving FP8 a 0.3–0.7 point accuracy cost and INT8 a 1.5–3 point cost. Those numbers came from an SEO blog and I could not trace them to any benchmark. **Cut.** I also claimed FP8 buys 30–50% throughput. Also wrong, by about 3×.

Here is what a real benchmark says. The vLLM team (with AWS and Red Hat AI) [published a full FP8 KV-cache validation](https://vllm.ai/blog/2026-04-22-fp8-kvcache) in April 2026 — deliberately using uncalibrated per-tensor scales as a worst case, reproducible with one flag:

**Accuracy.** At most 0.7 points degradation on Qwen3.5-27B reasoning, lowest recovery 99% on AIME25; 1–2 points on Qwen3-30B-A3B-Thinking. On long-context MRCR: 97–98% of baseline AUC at 128k for Llama-3.3-70B, and full recovery of aggregated AUC at 1M context on Qwen3.5-27B.

**Performance, measured under load** — not the number I made up:

```
Model            Median ITL    Output tok/s    Gain
──────────────────────────────────────────────────────
Llama-3.1-8B
  BF16           15.18 ms      450.3
  FP8            12.93 ms      517.5           +14.9%

gpt-oss-20b
  BF16            8.09 ms      831.6
  FP8 skip-SW     7.70 ms      871.8            +4.8%
```

The single-request ITL slope drops to 54% of BF16 for Llama-3.1-8B, pulling the decode break-even point down to about 7k tokens. Below that, BF16 is slightly faster.

**And here is the story that should be in every post on this subject and is in almost none.**

On Hopper, the FP8 Flash Attention 3 kernel lost accumulation precision at long context. On a 128k needle-in-a-haystack task, **FP8 accuracy dropped from 91% to 13%** — traced to imprecise FP32 accumulation in the Tensor Cores when the contraction dimension exceeds ~100K, the same hardware behaviour documented during DeepSeek-V3 training. A two-level accumulation fix brought it back to 89%, at the cost of register pressure that makes prefill slower for `head_dim = 256`.

Read that again. A config flag that looked accuracy-neutral was destroying long-context retrieval, and short-context evals would never have caught it. This is the same shape as the NIXL failure in part 5: the architecture was right, the layer underneath silently wasn't, and the dashboards you'd normally check said everything was fine.

**When not to use FP8**, per the same source: contexts under ~7k, `head_dim = 256` where prefill latency matters, models whose uncalibrated accuracy drops below 95%, and hybrid models with many small sliding-window layers (use `--kv-cache-dtype-skip-layers sliding_window`).

### Phase 6b — The cautionary tale: TurboQuant

I nearly published a paragraph hyping [TurboQuant](https://arxiv.org/abs/2504.19874) as 3-bit compression with zero accuracy loss. Almost every part of that was wrong, and correcting it produced a better section than the original.

**What the paper actually claims.** TurboQuant (Zandieh and Mirrokni of Google Research, Daliri of NYU, Hadian of Google DeepMind; arXiv 2504.19874, April 2025) randomly rotates input vectors to induce a concentrated Beta distribution, applies optimal scalar quantizers per coordinate, then corrects the residual with a 1-bit Quantized Johnson-Lindenstrauss transform. Its stated KV result: **absolute quality neutrality at 3.5 bits per channel, marginal degradation at 2.5.** Not 3 bits, not zero-loss. The polar-coordinate mechanism I'd attributed to it belongs to a different paper, [PolarQuant](https://arxiv.org/abs/2502.02617).

**What independent replication found.** vLLM ran [a comprehensive study across four models from 30B to 200B+](https://vllm.ai/blog/2026-05-11-turboquant) in May 2026. Their conclusion:

- **FP8 remains the best default** — 2× capacity, negligible accuracy loss, no throughput cost.
- **TurboQuant k8v4 offers no advantage over FP8** — 2.4× vs 2× savings isn't worth the latency and throughput penalty.
- **The aggressive 3-bit variants dropped up to 20 points** on AIME25 and LiveCodeBench, and up to 8 points on 64k long-context retrieval.
- All TurboQuant variants ran **below BF16 throughput** — 66–80% depending on variant.

The mechanism explains it. FP8 quantizes the attention computation itself on hardware-native Tensor Cores. TurboQuant compresses storage only and dequantizes back to BF16 to compute — so its overhead *grows with batch size*, the opposite of what you want.

**The exception worth knowing.** On a memory-starved Llama-3.3-70B deployment, BF16 P99 TTFT under burst exploded to ~17s as KV memory saturated and requests queued. TurboQuant variants stayed under 3.5s. FP8 hit ~1.3s. So compression genuinely rescues you from queueing — but FP8 rescues you better.

**The lesson generalises:** a compression ratio is not a serving win. Storage savings that require dequantization before compute can cost more than they save.

---

## The Competition: Shrink It vs. Delete It

Two escape routes from the same problem. (Framing: [Devansh](https://machine-learning-made-simple.medium.com/transformers-vs-mamba-vs-linear-attention-who-wins-long-context-f1dc8ceb5ede).)

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/kv-cache-escape-routes.png" alt="Branching diagram of Transformer-preserving techniques, Transformer-replacing architectures, and hybrids converging as the practical middle path" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
    <strong>Figure 3 — Two escape routes, and the convergence between them.</strong>
    Route A keeps exact recall and pays linear-but-slower growth. Route B eliminates
    growth and pays with lossy history. Route C is where production landed.
  </figcaption>
</figure>

**Route A — preserving.** Everything above. Exact token-level retrieval, memory that still grows.

**Route B — replacing.** Compress history into a fixed-size state and the cache stops growing. [Mamba](https://arxiv.org/abs/2312.00752) reports 4–5× higher inference throughput than a comparable Transformer, precisely because without a KV cache it can use much larger batches. Linear attention (RWKV, RetNet, GLA) drops softmax for kernel maps or decay-modulated products, with an O(1) recurrent inference form.

The limitation is informational, not engineering: a constant-size state cannot preserve arbitrary token-level recall over arbitrary length. The ["repeat after me" result](https://arxiv.org/abs/2402.01032) showed Transformers beating SSMs at copying specifically.

**Route C — hybrids.** Interleave a few attention layers with many recurrent ones. Jamba, Samba, Qwen3-Next, Nemotron Nano 2, MiniMax-Text-01, Granite 4.0. [vLLM now treats hybrid models as first-class citizens](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/), which is the clearest signal this left the research phase.

```
Route              KV growth        Exact recall   Maturity    Retrofittable?
──────────────────────────────────────────────────────────────────────────────
GQA / MQA          linear, /4-8     full           very high   no (train-time)
MLA                linear, /2.7-4.7 full           medium      no (train-time)
FP8 KV cache       linear, /2       full           high        YES
TurboQuant 4bit-nc linear, /3.4     modest drop    low         YES
Paged + prefix     linear, shared   full           very high   YES
Offload / tiering  bounded by tier  full           medium      YES
Pure SSM / linear  NONE             weak           low         no
Hybrid             sub-linear       good           rising      no
```

If you operate rather than train, read the last column first.

---

## The Systems Layer, and Where the Assumptions Break

**P/D disaggregation** splits the phases onto separate pools so each scales independently. Standard now across vLLM, SGLang, TensorRT-LLM, LMDeploy and NVIDIA Dynamo.

[Part 5](/llm-infrastructure/inference/2026/04/21/llm-d-pd-disaggregation.html) is my hard-won version: on a single GH200 with time-sliced pods there's no RDMA path, NIXL can't initialise, and the whole thing collapses into aggregated serving at half the memory budget. E2E p50 went 260ms → 1,690ms, with "No Data" rather than errors on every transfer metric.

**Tiering** extends this: GPU HBM → CPU DRAM → NVMe → object storage. [LMCache](https://arxiv.org/abs/2510.09665) reports 400 Gbps load bandwidth from CPU memory against 88 Gbps for vLLM's native offload — and the mechanism behind that 4.5× is worth more than the number.

It's transfer granularity. vLLM manages KV cache in pages of roughly 20–63 KB — 62.5 KB for Llama-3.1-8B-Instruct — and native offloading moves them one page at a time. Every transfer triggers a CUDA memcpy with metadata preparation beforehand and a completion signal afterward, so per-transfer overhead dominates when the units are that small. LMCache groups pages across multiple layers into configurable chunks, defaulting to 256 tokens, and moves those instead.

The supporting numbers make the point sharper. Prior work cited in that paper found transfer sizes must reach **16 MB to saturate** an eight-NIC 400 Gbps setup, and that only megabyte-range transfers achieve 75–80% of theoretical PCIe 5.0 bandwidth. Page-sized I/O leaves most of your interconnect on the floor.

This is the same lesson as the FP8 kernel and the TurboQuant dequantization overhead, a third time: **the layer beneath your abstraction determines whether the abstraction pays.**

### The truncation trap

One finding from LMCache's production deployments deserves its own heading, because it undoes the thing most of this post is about.

Many teams handle over-long inputs with a sliding window — truncate to keep only the most recent tokens. On real traces from one enterprise user, **prefix cache hit ratios dropped from roughly 85% to 45%** under truncation. Truncated inputs no longer match the prefixes of previously cached contexts, so the cache misses.

Sit with that against [part 4](/llm-infrastructure/inference/2026/04/19/llm-d-epp-prefix-cache-results.html). My 81.3% hit rate is in the same range as their pre-truncation 85%. A context-management policy applied one layer up, for entirely sensible reasons — fitting a context window, capping GPU memory — would have halved it, and nothing in the serving stack would have reported a fault. You'd just be paying twice for prefill.

The general form: **anything that dynamically adds or removes tokens from the front or middle of a context invalidates prefix reuse downstream.** If you're building agents, that includes context compaction, message pruning, and rolling summarisation. Append-only contexts cache; edited contexts don't.

A related and more cheerful finding from the same deployments: users were surprised by how *high* their hit rates were — one production system at 50% — because reuse isn't limited to fixed system prompts any more. Conversation histories in coding assistants, chat applications and RAG pipelines are all dynamically reusable context.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/kv-cache-bandwidth-cliff.png" alt="Log-scale bar chart of bandwidth per memory tier, showing the two-orders-of-magnitude drop from HBM to sustained PCIe" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
    <strong>Figure 4 — The bandwidth cliff that turns prefill memory-bound.</strong>
    PCIe 5.0's 64 GB/s is about 2% of HBM
    bandwidth, and measured sustained throughput is 15 GB/s — 23% of even that peak.
  </figcaption>
</figure>

### The three things that break

This is the part I'd lead with if you read nothing else. All three are from Meng et al.

**1. Prefill becomes memory-bound, at far lower ratios than the specs suggest.** Predicted κ_crit from peak hardware numbers: 14.3 for Llama-3.1-70B, 7.8 for Qwen3-235B. **Measured: 2 and 1.** The gap is sustained PCIe bandwidth — 15 GB/s against a 64 GB/s peak. Over 50% of execution time goes to transfers at κ_ratio as low as 1. At the extreme (Qwen, 65k cached, 64 new tokens), PCIe overhead reaches 86× the prefill computation — **99% of execution time.**

**2. Hardware specialisation stops paying.** P/D disaggregation routes prefill to expensive compute parts because prefill is compute-bound. When K ≫ T, prefill converges toward decode — both bandwidth-limited — and there's nothing left to specialise for. Measured GPU power draw on these workloads: **22–28% of TDP.** You're provisioning cooling and electrical capacity for compute that never happens.

**3. Token-budget scheduling under-counts memory.** vLLM charges a request its *new* prefill tokens. A request with 1,000 cached and 100 new tokens charges 100 against the budget while occupying VRAM for 1,100.

The consequence, from their B200 case study: a document Q&A request with 65k cached and 32 new tokens lets the scheduler fit **1.8 concurrent requests**, processing 57 tokens against a 4,000-token budget — **1.4% budget utilisation**. Measured across replayed workloads, ShareGPT averaged 4,064 scheduled tokens per iteration; NarrativeQA managed 532. VRAM exhausts long before the token budget does.

And the workloads really are that lopsided. Median κ_ratio: **100 for ShareGPT multi-turn chat, 5,000 for NarrativeQA document Q&A, 10,000 for FinQA.** Against measured κ_crit of 1–2.

This is the part that connects directly to [pd-ratio-coordinator](https://github.com/kraghavan/pd-ratio-coordinator): autonomously rebalancing prefill and decode replicas can't work correctly while the thing doing the rebalancing is fed accounting that ignores cached-context VRAM.

---

## Where This Goes

Extrapolation from here, anchored to the memory roadmap. Confidence stated.

### What we know about the hardware

Per-GPU HBM went from H100's 80 GB / 3.35 TB/s to Rubin's 288 GB across 8 HBM4 stacks at up to 22 TB/s — roughly 6.6× the bandwidth but only about 3.6× the capacity. NVLink 6 doubles to 3.6 TB/s per GPU; NVL72 reaches 20.7 TB HBM4 at ~1.6 PB/s aggregate.

Two complications worth stating plainly:

- NVIDIA reportedly lowered the HBM4 spec from 22 TB/s toward ~20 TB/s after suppliers missed the target.
- [TrendForce reported on 4 August 2026](https://www.trendforce.com/presscenter/news/20260804-13166.html) that since Q3 2026 NVIDIA has been evaluating alternatives to Rubin Ultra's original 12-Hi HBM4e baseline — 8-Hi HBM4e, 12-Hi HBM4, and 8-Hi HBM4 — with the final specification undetermined. The drivers: DRAM supply staying tight through 2027, and uncertainty in 12-Hi HBM4e validation and yield ramp.

**Attribution correction.** An earlier draft of this post said TrendForce reported an 8-Hi option yielding ~192 GB. TrendForce's release does not state that figure. The 192 GB number is **SemiAnalysis's** characterization of what 8-Hi HBM4 would yield, reported alongside TrendForce's configuration news; separately, The Information reported the lower-memory evaluation with TrendForce corroborating. Two sources, one claim, and I'd merged them.

The trend line is more striking than any single number. Rubin Ultra's memory spec has walked down from a 4-die 1 TB HBM4E 16-Hi configuration previewed at GTC 2025, through 12-Hi HBM4E, through a cancelled 4-die MCM, to the current 2-die package. Meanwhile TrendForce projects HBM bit shipments growing 50–60% year-over-year in 2027 and *still* falling short of demand.

### Prediction 1 — capacity pressure doesn't resolve; it moves up a tier. *(High confidence.)*

Architectures and context lengths expand to fill available HBM. Every past capacity increase was absorbed immediately. With the Rubin Ultra signal and DRAM shortage on top, per-token cache efficiency stays a first-order concern through 2028.

### Prediction 2 — faster accelerators make the memory wall *worse*, not better. *(High confidence — and this corrects my own earlier reasoning.)*

In a draft I argued that rising interconnect bandwidth makes tiering strictly more attractive. Meng et al.'s data says the opposite on the axis that matters: **compute scaling has outpaced interconnect scaling, so newer GPUs are more prone to PCIe bottlenecks.** B200 delivers 2.5× H100's effective compute while PCIe 5.0 gives only 2× PCIe 4.0's bandwidth. The result is B200's hardware factor at 13.5 against H100's 34 — **B200 goes memory-bound at 2.5× lower κ_ratio.**

Better interconnects help but don't rescue you. NVLink C2C's 900 GB/s (14× PCIe 5.0) raises κ_crit 5.3× — Qwen3-235B from 7.8 to 41.5, DeepSeek-V3 to 191. Document Q&A's median κ_ratio is 5,000. Still firmly memory-bound.

Their more ambitious proposal is unified CPU-GPU HBM on-package, which would raise κ_crit to 370 and 1,700 respectively. That's the first configuration in the paper where document Q&A approaches compute-bound — and it doesn't exist yet.

**The vendors appear to agree, and have said so.** TrendForce's read on the Rubin Ultra situation is that NVIDIA's primary objective for that generation is **increasing I/O speed, with expanding GPU shipment volume as a secondary priority** — and that if the HBM spec is cut, it'll be done by reducing stack layers. If HBM4e validates on schedule, I/O rises from Rubin's 8–11.7 Gbps to 14–16 Gbps; on HBM4 alone, 11–12 Gbps.

That's a vendor explicitly trading capacity for bandwidth. It's better evidence for this prediction than the capacity-regression framing I originally built the section on, because it isn't an inference from a supply constraint — it's a stated design priority.

### Prediction 3 — FP8 is the floor for a while; sub-4-bit stays niche. *(Medium-high confidence, revised down.)*

I previously predicted 4-bit becoming default the way FP8 did. The vLLM TurboQuant study makes me less confident. The decisive factor isn't bit-width, it's whether the format is **hardware-native in the attention computation**. FP8 wins because Tensor Cores compute in it. Any scheme that dequantizes to BF16 before computing pays an overhead that grows with batch size.

So: 4-bit becomes default *if and when* it becomes a native compute format (NVFP4 is a plausible path), and not before. Storage-only compression will keep losing on throughput regardless of how good the distortion bounds are.

### Prediction 4 — hybrids become default for new frontier models. *(Medium-high confidence.)*

A small number of attention layers recovers most of the SSM quality gap, and vLLM's first-class support removes the deployment friction. Pure SSMs don't win outright because the fixed-state recall limit is informational, and agentic workloads need exact retrieval over long histories.

**Falsifier:** a pure-SSM model matching Transformer multi-needle retrieval at 128k+. I don't expect it; if it happens I'm wrong rather than early.

### Prediction 5 — this is where the leverage is, and it isn't my idea

I want to be careful here, because in an earlier draft I presented this as my own insight and it isn't.

Meng et al. propose it directly: route by κ_ratio, sending high-κ_ratio requests to high-bandwidth hardware and compute-intensive prefill to cheaper compute-dense parts; power-cap high-κ_ratio servers to 200–300W since PCIe rather than compute limits them; replace FIFO token accounting with utilisation-aware scheduling that co-schedules complementary requests, with aging credits and admission control for fairness.

**My contribution is narrower and I'll state it as such.** I have measured data from the *other* side of this boundary. Part 4's 81.3% prefix hit rate is precisely the mechanism that drives κ_ratio up — successful reuse shrinks T. Part 5 is what happens when the transfer path required by that regime doesn't physically exist. The EPP was doing a primitive version of κ_ratio-aware routing without calling it that, and it worked.

What I'd want to instrument next isn't whether KV transfer happens, but **what it costs, which tier it came from, and whether the scheduler was charged correctly for it.** That's the gap between their analytical framework and a running system, and it's where a routing gateway earns its keep.

### Not predicting

- Whether MLA or hybrids "win" — not exclusive, and the timelines are genuinely uncertain.
- That efficiency reduces spend. Historically it's absorbed by longer contexts, more reasoning tokens, more agent turns.

---

## The Operator's Guide

**You can't change the model:**

1. **Turn on FP8 KV cache** if your contexts exceed ~7k. Measured: 2× capacity, ≤0.7–2 points accuracy, +14.9% throughput on Llama-3.1-8B under load. Skip sliding-window layers on hybrid models.
2. **Turn on prefix caching** — then understand that succeeding at it pushes you toward the memory-bound regime. Measure hit rate *and* resulting κ_ratio.
3. **Audit your context-management policy for prefix invalidation.** Sliding-window truncation halved one production deployment's hit rate, 85% → 45%, silently. Compaction, message pruning and rolling summarisation have the same effect. Append-only contexts cache; edited contexts don't.
4. **Compute your κ_crit before deploying offloading.** Use measured sustained bandwidth, not peak. If your workload's κ_ratio exceeds it by 100×, you are buying compute you will not use.
5. **Verify your interconnect before P/D disaggregation.** Part 5 exists so you don't learn this the way I did.

**Choosing a model:**

6. Below 32k context, KV architecture barely matters.
7. At 32k–128k, GQA is table stakes; MLA raises κ_crit 4.6× over comparable GQA MoEs — benchmark on your stack.
8. Above 128k with exact-retrieval needs: MLA + FP8 + prefix caching.
9. Above 128k where recall is approximate: hybrids deserve a look.

**Always:**

10. **Evaluate compression on multi-needle long-context retrieval at your real context length.** The 91%→13% FA3 regression is the proof: short-context evals would have shown nothing wrong.

---

## Open Problems

1. **Memory-aware scheduling.** Token budgets that count cached-context VRAM. Broken by construction today.
2. **κ_ratio-aware routing across heterogeneous fleets** with tier-cost modelling. Proposed by Meng et al.; not in production gateways.
3. **Native low-bit compute formats** — not storage compression that dequantizes before attention.
4. **Empirical MLA characterisation under offloading.** Meng et al. explicitly defer it as future work after their DeepSeek-V2 attempt failed to isolate transfer time.
5. **Hybrid cache management.** PagedAttention assumes uniform KV blocks; hybrids have attention layers *and* recurrent state with different lifecycles.

---

## What This Means for the Series

Parts 1–5 went bottom-up: what inference does, measuring it on an M4 Mac Mini, deploying llm-d on a GH200, EPP prefix caching, then the P/D attempt that taught me RDMA is a correctness condition.

This post is the map I wish I'd had before part 3 — and writing it changed what I think part 7 should be. The next experiment isn't just "P/D with real RDMA." It's measuring κ_ratio on a realistic workload, computing κ_crit for the actual hardware, and checking whether the scheduler's token accounting matches its VRAM consumption.

That's a smaller experiment than a two-node RDMA build, and it would tell me more.

---

## Sources and Corrections

This post went through a verification pass that killed several claims from its first draft. Recording that, because the corrections are more useful than the original text.

**Corrected:**
- An FP8-vs-INT8 accuracy table sourced from an SEO blog — **removed**. No traceable benchmark. The INT8 comparison has no credible KV-cache-specific source I could find.
- "FP8 gives 30–50% throughput" — **wrong by ~3×**. Measured: 14.9% (Llama-3.1-8B) and 4.8% (gpt-oss-20b).
- "70–90% of VRAM at 1M tokens" — **removed**, untraceable.
- TurboQuant as "3-bit, zero accuracy loss, polar transformation, ICLR 2026" — **wrong on nearly every detail**. Paper claims neutrality at 3.5 bits; the mechanism is random rotation plus a 1-bit QJL residual; polar transformation is PolarQuant, a different paper; the venue claim came from an unofficial PyPI clone. Independent vLLM replication found the aggressive variants drop up to 20 points.
- "MLA benefits can't be demonstrated because serving implementations are immature" — **overstated**. Source says they hit implementation-specific overheads evaluating DeepSeek-V2 that prevented clean PCIe isolation.
- "Rising bandwidth makes tiering more attractive" — **reversed**. Compute is outpacing interconnect; newer GPUs go memory-bound sooner.
- Prediction 5 — **reattributed** to Meng, Lee & Wang, who proposed κ_ratio-aware routing and utilisation-aware scheduling.
- The preserving/replacing framing — **credited** to Devansh.
- "TrendForce reported ~192 GB for Rubin Ultra" — **misattributed**. TrendForce reported the four-configuration evaluation and the supply drivers; the 192 GB figure is SemiAnalysis's characterization of 8-Hi HBM4.

**Primary sources read in full:** Meng, Lee & Wang, arXiv 2601.19910v1 (preprint, MLSys submission format, 16 Dec 2025); LMCache, arXiv 2510.09665v2 (Liu, Yao, Cheng et al., Tensormesh + UChicago, 5 Dec 2025); the vLLM FP8 KV-cache post (Apr 2026); the vLLM TurboQuant study (May 2026); TrendForce press release 20260804-13166.

**Read as abstract only:** TurboQuant (arXiv 2504.19874), PolarQuant (arXiv 2502.02617), Mamba (arXiv 2312.00752), PagedAttention (arXiv 2309.06180).

**Every substantive claim in this post now traces to a source you can open.** Where a number comes from a preprint, a vendor blog, or a single analyst firm, I've said so inline. Where I could not trace something, it was removed rather than hedged.

**My own measurements:** the 81.3% prefix hit rate and the TTFT/ITL/E2E figures, from parts 4 and 5, on a Lambda Labs GH200 with Qwen3-0.6B. Small model, single GPU — don't generalise them to production topologies.

---

*No new experiments in this one — a landscape post built on the measurements from parts 4 and 5, with a verification pass that reshaped it. Platform engineer with 11+ years in distributed systems going deep on LLM serving infrastructure.*

*[GitHub](https://github.com/kraghavan) · [LinkedIn](https://linkedin.com/in/karthikaraghavan)*
