---
layout: post
title: "What Is LLM Inference, Really? A Deep Technical Walkthrough"
date: 2026-04-14
categories: [llm-infrastructure, inference]
tags: [vllm, inference, ttft, tpot, kv-cache, attention, tokenization, sre, transformers]
series: "LLM Inference from First Principles (A Re-introduction)"
series_part: 1
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: >
  An Engineer's annotated tour through what actually happens when you hit send —
  from bytes to tokens to embeddings to attention to the word your model finally spits out.
  No skipped steps. No "and then magic happens."
---

Let me be honest with you. When I started working on LLM infrastructure, I had eleven years of distributed systems experience. I knew Kafka, Kubernetes, Prometheus. I could debug a partition rebalance in my sleep.

And yet the first time someone asked me *what actually happens during inference*, I said something like "the model reads the prompt and generates tokens." Which is technically true the same way "a database reads your query and returns rows" is technically true — accurate, useless, and deeply embarrassing for someone drawing a principal engineer's salary.

This post is what I wish I'd had on day one. We're going to walk through the entire inference pipeline — from the moment your request arrives to the moment you see text on screen — with real examples, honest explanations of where the performance goes, and enough detail that you can actually reason about production problems.

No "and then the transformer does its thing." No skipped steps. Strap in.

---

<figure style="max-width:480px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/gemini-generated-llm-inference-pipeline.png" alt="LLM Inference Pipeline" style="width:100%;">
</figure>

---

## 1. What Is Inference?

**Training** is where you take a massive dataset, run it through a model millions of times, and slowly adjust billions of numerical weights until the model gets good at predicting the next word. Training is done once (or occasionally). It costs millions of dollars in GPU-hours and requires a team of researchers.

**Inference** is what happens afterward, every time someone uses the model. It's the model *using* those learned weights to respond to new input. No weights change. No learning happens. It's pure forward-pass computation.

Think of it like this: training is baking the bread. Inference is slicing it and serving it to customers. The bread (weights) is done. The kitchen (inference engine) just has to plate it fast enough that the queue doesn't back up to the street.

The inference engine is the runtime that takes the frozen model weights and executes them against an input. The same weights can run on Ollama, vLLM, TensorRT-LLM, or TGI — and get meaningfully different performance from each. The weights don't change. The execution strategy does.

This distinction matters operationally: **inference is not a solved problem**. Serving a model efficiently at scale is a full engineering discipline.

---

## 2. The Three Phases: A Map Before the Territory

Every inference request goes through three broad phases. They are not equally expensive, not equally parallelizable, and not equally friendly to your p99 latency.

```
┌──────────────────────────────────────────────────────────────────┐
│  TOKENIZATION  │  Prefill (Prompt Processing)   │ Decode         │
│    (CPU)       │           (GPU, compute)       │ Loop(GPU, Mem) │
│                │                                │                │
│                │                                │ current →      │
│  Text → IDs    │  Embed → Position → Attention  │ next token     │
└──────────────────────────────────────────────────────────────────┘
     Fast          Scales with prompt length        Slow
```

- **Tokenization**: Split the text into token IDs the model understands. CPU-bound. Fast.
- **Prefill**: Process the entire prompt through the model. GPU compute-bound. Scales with prompt length.
- **Decode**: Generate output tokens one at a time. GPU memory-bound. Runs in a loop until done.

Each phase has its own bottleneck. Let's go through them one by one.

---

## 3. Tokenization: Chopping Text Into Numbers

Before a single GPU operation happens, your text has to be converted into a format the model can work with: a sequence of integers called token IDs.

A token is not a character, and it's not a word. It's a chunk of text that appears frequently enough in the training corpus to deserve its own ID. There are several ways to build this vocabulary — WordPiece (used by BERT), Unigram (used by SentencePiece), and others — but the dominant approach in modern LLMs is Byte Pair Encoding (BPE): a compression algorithm that iteratively merges the most common pairs of characters or subwords into single tokens until it reaches a target vocabulary size.

The result is a vocabulary of roughly 32,000–128,000 tokens, each with a corresponding integer ID. The model never sees your text — it sees a list of numbers.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/encoding-bpe.png" alt="Tokenization with BPE" style="width:100%;">
</figure>

Take our example prompt: `"The cat sat"`

After tokenization, this becomes something like:
```
"The"  → 1026
" cat" → 5992
" sat" → 3290
```
Token IDs: `[1026, 5992, 3290]`

Note the space before "cat" and "sat" — it's part of the token. Tokenizers care about whitespace because it affects meaning and frequency.

### Is Tokenization CPU-Bound?

Yes. The tokenizer is usually written in Rust (HuggingFace's `tokenizers` crate) or C++ for exactly this reason. For most requests it's fast enough to be invisible — microseconds for a short prompt.

Where it bites you: **very long documents** fed to batch processing jobs. A 100,000-token context requires processing 100,000 token lookups. It's still fast relative to GPU work, but it's the one step in the pipeline running on CPU that you can't just throw more GPU at.

**How it's improved:** Parallelizing tokenization across CPU cores for batch workloads. Or — and this is the real fix — **not re-tokenizing the same content repeatedly**. If you have a shared system prompt you send to every request, tokenizing it once and caching the result is free latency.

---

## 4. Prefill: The Model Reads Your Prompt

Now we have token IDs. The model needs to turn those IDs into something it can reason about. This is **prefill** — the model processing the entire prompt in one shot.

Prefill has two sub-steps that are easy to conflate: **embedding lookup** and the actual **transformer forward pass**. Let's take them in order.

### The Embedding Matrix

Every token ID maps to a high-dimensional vector of floating-point numbers called an **embedding**. These vectors live in the model's embedding matrix — a giant lookup table with one row per vocabulary token and one column per embedding dimension.

For a model with a 32,000-token vocabulary and 4,096 embedding dimensions, this matrix has shape `[32000, 4096]`. At 16-bit float precision, that's about 256MB just for the embedding layer.

Our example `[1026, 5992, 3290]` becomes:

```
Token ID 1026 → embedding row 1026 → [0.12, -0.43, 0.81, ..., 0.07]  (4096 values)
Token ID 5992 → embedding row 5992 → [-0.34, 0.91, 0.12, ..., -0.22] (4096 values)
Token ID 3290 → embedding row 3290 → [0.67, 0.05, -0.88, ..., 0.44]  (4096 values)
```

I'm simplifying to 8 dimensions here so this fits on a page. In reality it's 4,096 or 8,192 dimensions depending on the model.

```
Simplified (3D instead of 4096D), just to show the shape:

"The"  → [0.12, -0.43,  0.81]
" cat" → [-0.34,  0.91,  0.12]
" sat" → [0.67,  0.05, -0.88]

Shape: [3 tokens × 3 dims] = a matrix of floats
```

These vectors aren't random. They're the result of training — the model has learned that "cat" and "dog" live close together in this space, and "cat" and "quantum mechanics" are far apart. The geometry encodes semantic meaning.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/embedding-lookup-process.png" alt="Embedding Lookup Process" style="width:100%;">
</figure>

### How Do the Model Weights Help Here?

The embedding matrix IS the model weights, specifically. The 10GB (or 40GB, or 70GB) file you download — the GGUF or safetensors file — contains all the weight matrices the model learned during training. The embedding lookup is literally indexing into one of those weight matrices by row number.

When you run inference, you're not computing anything creative. You're doing matrix math against frozen numbers that were tuned over millions of training iterations.

---

## 5. Positional Embeddings: Teaching the Model About Order

Here's a problem: the embedding lookup is a table lookup. It doesn't care that "cat" is token 2 and "sat" is token 3. Two requests with the same tokens in different orders would produce identical embeddings.

But order matters enormously. "The cat sat on the dog" and "The dog sat on the cat" have the same tokens and very different meanings.

**Positional embeddings** solve this by adding a position-aware vector to each token's embedding. The model learns that "token at position 1" feels different from "token at position 5," even if the token ID is the same.

### How Is It Calculated?

There are two main approaches:

**Sinusoidal (original Transformers paper):** Compute a fixed sine/cosine wave pattern based on position and dimension index. Deterministic, no learned parameters.

**RoPE (Rotary Position Embedding):** Used by Llama, Qwen, Mistral, and most modern models. Instead of adding a vector, it *rotates* the query and key vectors by an angle proportional to position. The result: the dot product between two token representations naturally captures their relative distance. Elegant, and generalizes better to longer contexts than the training data.

Continuing our example. After adding positional information:

```
"The"  at position 0: [0.12, -0.43, 0.81] + pos(0) → [0.15, -0.40, 0.84]
" cat" at position 1: [-0.34, 0.91, 0.12] + pos(1) → [-0.31, 0.88, 0.09]
" sat" at position 2: [0.67, 0.05, -0.88] + pos(2) → [0.65, 0.03, -0.86]
```

The position vectors are small adjustments. Their real value is that when attention is computed later, the model can tell whether two tokens are adjacent or 200 positions apart.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/positional-embeddings-diagram.png" alt="Positional Embeddings Diagram" style="width:100%;">
</figure>

### CPU Bottleneck in Prefill?

Embedding lookup and positional encoding are fast operations. The real CPU bottleneck in prefill is less about these steps and more about **data movement**: loading the right weight tensors from CPU RAM to GPU VRAM before the transformer forward pass can begin.

For very large models that don't fully fit in VRAM, the CPU-GPU transfer becomes the bottleneck — you're constantly paging weight blocks in. This is why model quantization matters: a 4-bit quantized model uses less VRAM, fits entirely on GPU, and eliminates this transfer overhead. More on that in a moment.

---

## 6. The Transformer Layers: Where the Real Work Happens

After embedding + positional encoding, we have a matrix of shape `[sequence_length × embedding_dim]`. This matrix now passes through N transformer layers — 32 layers for Llama-3.2-3B, 80 layers for Llama-3.1-70B.

Each layer applies:
1. **Self-attention**: every token looks at every other token and decides what's relevant
2. **Feed-forward network (FFN)**: each token's representation is independently transformed

This is where the model's "reasoning" happens — and where most of the GPU compute goes during prefill. All tokens are processed in parallel within a layer, making prefill compute-bound. More tokens = more compute = higher TTFT.

We'll cover the attention mechanism in detail in section 9. First, let's see what comes out.

---

## 7. Decoding: One Token at a Time, Forever

After prefill, the model produces its first output token. Then it produces another. Then another. Each token depends on all previous tokens. This is the **decode loop**.

Here's what makes decode fundamentally different from prefill: **you can't parallelize it**. Token N can't be computed until token N-1 exists. It's inherently sequential.

Let's walk through two steps with our example. Our prompt was "The cat sat" and let's say the model is going to output "on the mat."

### Decode Step 1: Predicting "on"

After prefill, we have KV cache entries for "The", " cat", " sat" (we'll explain KV cache shortly). Now:

1. The model takes the last token's representation (" sat") and runs it through the transformer layers
2. At each layer, attention is computed between " sat" and all previous tokens via the KV cache
3. The final layer outputs a vector of size `[vocabulary_size]` — one score per possible next token. This is called the **logits** vector.
4. The logits are converted to probabilities via softmax
5. A token is sampled from this distribution (more below)
6. Result: token ID for " on" → `" on"` is emitted as the first output token

```
KV cache: ["The", " cat", " sat"]
Current:  " sat" (last input token)
Attention: " sat" attends to "The", " cat", " sat"
Output logits: [0.001, 0.003, ..., 0.45 (" on"), ..., 0.002]
Sample: " on" ✓
```

### Decode Step 2: Predicting "the"

Now " on" has been generated. We add it to context:

1. Embed token " on" → one new embedding vector (just *one* token, not the whole sequence)
2. Add positional embedding for position 4
3. Run through transformer layers, attending to KV cache for ["The", " cat", " sat", " on"]
4. Output logits → sample → " the"

```
KV cache: ["The", " cat", " sat", " on"]  ← one entry added
Current:  " on" (new last token)
Attention: " on" attends to all four previous tokens
Output logits: [..., 0.67 (" the"), ...]
Sample: " the" ✓
```

And so it continues: " mat" → "." → `<end>` token → stop.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/key-value-cache-growth.png" alt="Key Value Cache Growth" style="width:100%;">
</figure>

### The Sampling Step (Where Creativity Lives)

The logits give you a probability distribution. How you sample from it is where temperature, top-k, and top-p come in:

- **Greedy (temperature=0)**: always pick the highest probability token. Deterministic. Boring for creative tasks, good for code.
- **Temperature > 1**: flatten the distribution. More randomness, more surprising outputs, more hallucinations.
- **Temperature < 1**: sharpen the distribution. More conservative, more predictable.
- **Top-k**: only sample from the top K most probable tokens. Ignores the long tail.
- **Top-p (nucleus sampling)**: only sample from the smallest set of tokens whose cumulative probability exceeds p. Adaptive — sometimes that's 2 tokens, sometimes 50.

This step is trivially cheap computationally but has enormous impact on output quality. As an SRE, you don't usually tune this — but you will get bug reports when someone's temperature=2.0 config makes the model output Shakespeare from a JSON API endpoint.

---

## 8. Why Memory Is the Decode Bottleneck

Here's the thing about decode that makes it hard to optimize: on every single decode step, the model needs to run attention against **all previous tokens**. Not a summary of them. All of them. Via the KV cache.

For a 1000-token conversation, every decode step reads 1000 rows of KV tensors from GPU memory. For a 32-layer model, that's 32 reads of a large tensor. For a 7B model, the KV entry for a single token at a single layer is tens of kilobytes.

The GPU's compute cores can execute these operations fast. But they're waiting on HBM (High Bandwidth Memory) to deliver the data. The memory bus saturates before the compute does.

**The GPU is memory-bound during decode, not compute-bound.**

This is why adding more CUDA cores doesn't help decode performance as much as adding more memory bandwidth. It's why H100s are faster than A100s for serving despite similar compute specs — the memory bandwidth jump matters more for decode than the FLOP count.

A rough intuition: during prefill, GPU utilization is high and memory is barely stressed. During decode, GPU compute is mostly idle and the memory bus is screaming.

---

## 9. KV Cache: The Most Important Data Structure in Inference

We keep mentioning the KV cache. Let's make it concrete.

During the attention step in each transformer layer, every token produces two vectors: a **Key** (K) and a **Value** (V). These are used in attention computation: other tokens use your Key to decide how much to attend to you, and then use your Value to extract information from you.

During decode, token N needs to compute attention against tokens 1 through N-1. If we recomputed K and V for all previous tokens on every step, we'd be doing O(N²) work per decode step. That's catastrophic.

The KV cache solves this: after computing K and V for a token, we **store** them. On the next decode step, we only compute K and V for the new token and look up the rest from cache.

```
After prefill of "The cat sat":
KV cache = {
  layer_0: { K: [k_The, k_cat, k_sat], V: [v_The, v_cat, v_sat] }
  layer_1: { K: [...], V: [...] }
  ...  (32 layers total)
}

After generating " on":
KV cache = {
  layer_0: { K: [k_The, k_cat, k_sat, k_on], V: [v_The, v_cat, v_sat, v_on] }
  ...
}
← one new entry appended per layer per decode step
```

The cache grows with every token generated. When the KV cache fills GPU memory:
- New requests queue (they have nowhere to store their KV tensors)
- Long requests get partially evicted and have to recompute (latency spike)
- In the worst case: OOM crash

KV cache occupancy is the single most important resource to monitor in a serving system. It determines how many concurrent requests you can serve and how long those requests can be. When `vllm:kv_cache_usage_perc` starts approaching 0.9, you're about to have a bad time.

### Prefix Caching: Free Speedups

If two requests share the same system prompt, their KV cache entries for that prefix are identical. Prefix caching stores those entries once and reuses them.

In practice on my M4 Mac Mini:
```
Without prefix cache (cold):  TTFT 753ms
With prefix cache (warm):     TTFT 367ms
Savings:                       51% reduction in TTFT
Zero model changes required.
```

This is why your production system prompt should be at the beginning of every request, and why it should be identical byte-for-byte across requests. Drift in the system prompt = cache misses = higher TTFT = sad users.

---

## 10. Attention: The Mechanism That Makes It Work

Let's go one level deeper into what happens at each transformer layer. Attention is the core operation. Everything else is bookkeeping.

### Self-Attention in Plain English

For each token, attention computes: *"which other tokens should I be paying attention to, and by how much?"*

It does this via three learned projections of each token's embedding:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I offer?"
- **Value (V)**: "what information do I contain?"

For each token, you compute its dot product with every other token's Key. High dot product = high attention score = attend more to that token. The scores are normalized via softmax, then used to weight-sum all the Value vectors.

Example with "The cat sat":

```
Token: " sat"
Q_sat · K_The  = 0.8  → attend heavily to "The"
Q_sat · K_cat  = 0.9  → attend most to " cat" (makes sense)
Q_sat · K_sat  = 0.3  → attend a little to itself

Attention weights after softmax: [0.35, 0.55, 0.10]

Output = 0.35 × V_The + 0.55 × V_cat + 0.10 × V_sat
```

The output for " sat" is now a blend of information from all tokens, weighted by relevance. After 32 such layers, the model has a rich, contextualized representation of every token in the sequence.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/attention-mechanism.png" alt="Attention Mechanism" style="width:100%;">
</figure>


### Paged Attention: Virtual Memory for KV Cache

Standard attention assumes the KV cache is one contiguous block of memory per request. This is wasteful: you have to pre-allocate the maximum possible sequence length upfront, and if the request ends early, that memory is wasted until the request completes.

**PagedAttention** (the key innovation in vLLM) borrows from OS virtual memory. Instead of one contiguous block, KV cache is stored in **fixed-size pages** (blocks) that can be non-contiguous in physical GPU memory. A page table maps logical token positions to physical blocks.

```
Without PagedAttention:
  Request A: [KKKKKKKKKK........] ← pre-allocated 20 slots, using 10, wasting 10
  Request B: [KKKKKK............] ← pre-allocated 20 slots, using 6, wasting 14

With PagedAttention:
  Block pool: [B1][B2][B3][B4][B5][B6][B7][B8]...
  Request A:  blocks B1, B3, B7 (non-contiguous, allocated on demand)
  Request B:  blocks B2, B4    (uses only what it needs)
```

The result: **much higher memory utilization**, more concurrent requests, less waste. Prefix cache blocks can be shared between requests with identical prefixes — only one copy of the system prompt's KV entries needed, regardless of how many requests use it.

PagedAttention is why vLLM typically serves 2-4x more concurrent requests than a naive implementation on the same hardware.

---

## 11. The GGUF File: What's Actually in That 10GB Download?

When you `ollama pull mistral` or download a model from HuggingFace, you're downloading the **model weights** — billions of floating-point numbers that encode everything the model learned during training.

**GGUF** (GPT-Generated Unified Format, formerly GGML) is the file format used by llama.cpp and Ollama. **Safetensors** is what HuggingFace uses. Both are essentially: metadata + serialized tensor blobs.

What's inside a 7B parameter model file:

```
GGUF file structure (simplified):
├── Header
│   ├── Model architecture (LlamaForCausalLM)
│   ├── Vocabulary (32000 tokens + their embeddings)
│   ├── Context length (4096, 8192, etc.)
│   └── Hyperparameters (n_layers, n_heads, etc.)
│
└── Weight tensors:
    ├── token_embeddings        [32000 × 4096]   ← the embedding matrix
    ├── layer.0.attention.q     [4096 × 4096]    ← Query projection weights
    ├── layer.0.attention.k     [4096 × 4096]    ← Key projection weights
    ├── layer.0.attention.v     [4096 × 4096]    ← Value projection weights
    ├── layer.0.attention.out   [4096 × 4096]    ← Output projection
    ├── layer.0.ffn.up          [4096 × 11008]   ← Feed-forward up
    ├── layer.0.ffn.down        [11008 × 4096]   ← Feed-forward down
    ├── ... × 32 layers
    └── output_norm + lm_head   [32000 × 4096]   ← Final projection to logits
```

**How weights are used during inference:**

- `token_embeddings`: the lookup table for step 4 — index it by token ID to get the embedding vector
- `layer.N.attention.q/k/v`: matrix-multiply the token embedding to get Q, K, V vectors
- `layer.N.ffn.*`: matrix-multiply through the feed-forward network
- `lm_head`: the final matrix-multiply that projects from embedding space to vocabulary space, giving you the logits

Every inference request is essentially: look up values from these weight matrices, multiply matrices together, repeat 32 times. The weights never change. You're doing very expensive matrix math against a very large lookup table.

### Quantization: Same Model, Smaller File

The same weight matrices can be stored at different numeric precisions:

| Precision | Bits per weight | 7B model size | Quality loss |
|---|---|---|---|
| FP32 | 32 | ~28GB | None (reference) |
| FP16 / BF16 | 16 | ~14GB | Minimal |
| INT8 | 8 | ~7GB | Small |
| INT4 (Q4_K_M) | 4 | ~4GB | Noticeable at edge cases |
| INT2 | 2 | ~2GB | Significant |

Quantization converts those 16-bit floats to lower-precision integers with a small scale factor. It's lossy compression — but the quality loss is often acceptable, and the memory savings are dramatic. A 7B model in Q4 fits on a MacBook Pro; in FP16 it needs an A100.

For operations: **lower precision = smaller tensors = faster memory transfers = better decode throughput**. INT4 models often have 20-40% better TPOT than their FP16 counterparts because the memory bottleneck loosens.

---

## 12. Continuous Batching: The Throughput Unlock

Early inference servers were naive: accept a batch of requests, run them all through the model together, return all responses. Simple.

The problem: requests finish at different times. Short requests had to wait for long requests to complete before the next batch could start. GPU utilization looked like a sawtooth wave.

**Continuous batching** (also called iteration-level scheduling) fixes this. Instead of batching at the request level, the inference engine batches at the **decode step** level. Every iteration, it assembles the currently active tokens — some mid-generation, some just starting — into a single GPU operation.

When a request finishes, its slot is immediately available for a new request. When a new request arrives, it joins the active batch at the next iteration rather than waiting for the next batch boundary.

The result: GPU utilization stays high, latency for new requests is low, and throughput scales with the number of concurrent requests the KV cache can support — not with some fixed batch size parameter you tuned last Tuesday.

vLLM, TGI, and TensorRT-LLM all implement continuous batching. Ollama does not (as of early 2025). This is one of the primary reasons vLLM serves at 2x the throughput of Ollama under concurrent load.

---

## 13. The Metrics That Matter

Now that you know what's happening, the metrics become obvious rather than mysterious.

<figure style="max-width:900px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/grafana-metrics.png" alt="Grafana Metrics Dashboard" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">Real metrics from a running llm-d deployment: TTFT p50 at 15ms, ITL p50 at 5ms, KV cache prefix hit rate at 80.6% — exactly the four numbers you should have on your wall during an incident.</figcaption>
</figure>

### TTFT — Time to First Token

**Definition:** Wall clock time from request submission to first output token.

**What it captures:** Prefill time + queuing time. If your GPU is busy, TTFT absorbs the wait.

**PromQL:**
```
histogram_quantile(0.95,
  sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le)
)
```

**SLO guidance:** < 500ms for interactive chat. < 2s is tolerable. > 5s and users think it's broken.

**Root causes when high:**
- Long prompts (expected — prefill scales with length)
- GPU under heavy load, requests queuing
- Insufficient prefill capacity (in P/D disaggregated setups)

---

### ITL — Inter-Token Latency (aka TPOT)

**Definition:** Average time between consecutive output tokens during decode. Inverse of token generation speed.

**What it captures:** Decode throughput per active request. Memory bandwidth is the primary lever.

**PromQL:**
```
histogram_quantile(0.95,
  sum(rate(vllm:time_per_output_token_seconds_bucket[2m])) by (le)
)
```

**SLO guidance:** < 30ms is fast, streaming feels smooth. > 100ms and you notice the typewriter effect.

**Root causes when high:**
- Too many concurrent requests (KV cache reads competing)
- Large model + small GPU memory bandwidth
- KV cache approaching capacity

---

### KV Cache Hit Ratio

**Definition:** Fraction of prompt tokens whose KV vectors were served from prefix cache vs recomputed.

**What it captures:** Effectiveness of prefix caching. High hit ratio = lower TTFT for repeated system prompts.

**PromQL:**
```
vllm:gpu_prefix_cache_hit_rate
```

**Target:** > 0.5 for most chat workloads with consistent system prompts. Near 0 means your prompts are fully unique (batch processing, no shared prefix).

---

### KV Cache Usage

**Definition:** Fraction of total KV cache capacity currently occupied.

**PromQL:**
```
vllm:gpu_cache_usage_perc
```

**Alert threshold:** > 0.85. At 0.9+, vLLM starts evicting in-progress requests, causing recomputation and latency spikes. At 1.0, new requests queue entirely.

---

### Scaling Strategy by Traffic Shape

This is where the SRE work actually lives:

| Traffic Pattern | Symptom | Action |
|---|---|---|
| Many short prompts, many users | TTFT fine, ITL rising, KV cache full | Scale out (more replicas), or reduce `max_model_len` |
| Few long prompts, long outputs | TTFT high, ITL high, KV cache full fast | Larger GPU memory, or P/D disaggregation |
| Bursty traffic, idle baseline | P99 TTFT spikes, P50 fine | Horizontal scaling + request queuing |
| Consistent system prompt across requests | High TTFT on cold start only | Enable prefix caching (already default in vLLM >= 0.4) |
| Mixed short and long context | Unpredictable KV usage | Set per-request `max_tokens` limits strictly |

**The strategic insight:** short prompt, many concurrent users → **decode is the bottleneck**, optimize for memory bandwidth and parallelism across requests. Long context, few users → **prefill is the bottleneck** and KV cache pressure is the constraint; P/D disaggregation helps by giving prefill its own GPU.

---

## 14. Where Does the Time Actually Go?

After running these experiments on real hardware, here's the honest answer:

```
Typical inference request (short prompt, moderate load):

Tokenization:          ~0.5ms   (CPU, negligible)
Embedding lookup:      ~1ms     (GPU memory read)
Prefill (32 layers):   ~40ms    (GPU compute, scales with prompt length)
First token decode:    ~3-5ms   (GPU memory read, KV cache write)
...each subsequent token: ~3-5ms

Total TTFT:            ~45ms under no load
Total TTFT at p95:     300-500ms under load (queuing dominates)
```

**Where load makes it worse:**

1. **Queuing before prefill starts**: your request sits behind other long prefills. TTFT absorbs the entire queue wait.
2. **KV cache contention during decode**: more concurrent requests = more KV cache reads per step = higher ITL for everyone.
3. **Memory fragmentation**: without PagedAttention, wasted KV cache blocks reduce effective concurrency.

**What's predicted to improve this:**

- **Speculative decoding**: a small "draft" model generates 4-5 tokens speculatively; the large model verifies them in one forward pass. If accepted, 4 tokens for the price of ~1.5 forward passes. Reduces ITL dramatically under low concurrency, hurts at high concurrency (wasted draft compute).
- **P/D disaggregation**: dedicated prefill GPUs handle prompt processing, dedicated decode GPUs handle generation. Eliminates the resource contention between phases. Requires fast interconnect (NVLink or RDMA) for KV transfer.
- **Flash Attention 3**: kernel-level optimization that keeps attention computation in SRAM longer, reducing HBM reads. Already default in vLLM on H100.

---

## 15. The Inference Engines Worth Knowing

Not all inference engines are created equal, and the right tool depends on your constraints.


<figure style="max-width:800px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/llm-inference/llm-inference-engine-comparison.png" alt="LLM Inference Engines Comparison" style="width:100%;">
</figure>

### Ollama
- **Origin:** Open source, community  
- **Strengths:** Dead simple to run, supports Apple Silicon natively, GGUF format  
- **Weaknesses:** No continuous batching (as of early 2025), worse throughput under concurrent load  
- **When to use:** Local development, single-user experiments, quick model testing  
- **When not to use:** Serving more than one user concurrently
- **GitHub:** [github.com/ollama/ollama](https://github.com/ollama/ollama)

### vLLM
- **Origin:** UC Berkeley, now a major open-source project with significant industry contributors  
- **Strengths:** PagedAttention, continuous batching, Prometheus metrics out of the box, P/D disaggregation via llm-d  
- **Weaknesses:** More complex setup, CUDA-first (Apple Silicon support via vllm-metal is experimental)  
- **When to use:** Production serving, multi-user, research with real load  
- **When not to use:** You just want to run one model locally and don't want to think about it
- **GitHub:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)  


### TGI (Text Generation Inference)
- **Origin:** HuggingFace  
- **Strengths:** First-class support for new HuggingFace models, FlashAttention, tensor parallelism  
- **Weaknesses:** Slower to adopt innovations than vLLM, somewhat opinionated config  
- **When to use:** You're already in the HuggingFace ecosystem and want good defaults  
- **GitHub:** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)  


### TensorRT-LLM
- **Origin:** NVIDIA  
- **Strengths:** Best possible performance on NVIDIA hardware, optimized kernels, inference graph compilation  
- **Weaknesses:** NVIDIA-only, complex setup, compiled engines are model+hardware-specific (can't move them)  
- **When to use:** You have a fixed model, fixed NVIDIA hardware, and need maximum performance  
- **When not to use:** You want flexibility, you're running experiments, or you don't own your hardware
- **GitHub:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)  


### The META / Research Options
- **llama.cpp** — [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp): The CPU-first runtime. Runs quantized models on CPU, reasonably fast, the ancestor of Ollama.
- **ExLlamaV2** — [github.com/turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2): Highly optimized for RTX GPUs specifically. EXL2 quantization format is more sophisticated than GPTQ or AWQ — per-layer bit allocation instead of uniform quantization.
- **MLC-LLM** — [github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm): Cross-platform (compiles to CUDA, Metal, Vulkan). Good for deploying to diverse hardware.

---

## 16. The Languages Behind It All

If you open the source code of a modern inference engine, here's what you'll find:

**Python**: The top layer. API server, request handling, scheduling logic, metric collection. vLLM's scheduler and OpenAI-compatible API are Python. This is also where most bugs live.

**CUDA (C++)**: The performance layer. Attention kernels, memory management, GPU operations. Flash Attention is CUDA. PagedAttention's physical block management is CUDA. If Python is the restaurant, CUDA is the kitchen.

**Rust**: The fast utilities layer. HuggingFace's `tokenizers` library is Rust — because tokenizing millions of requests fast matters. NIXL (the KV cache transfer layer in llm-d) has Rust/C++ components. Growing presence in inference tooling.

**Go**: The orchestration layer. Kubernetes operators, control plane tooling, health checks, routing logic. If you're writing infrastructure *around* inference — operators, routers, schedulers — Go is where the work happens.

**C++ (non-CUDA)**: llama.cpp is pure C++ with optional CUDA/Metal backends. TensorRT-LLM has heavy C++ in the engine.

**For SREs/DevOps:** You live in Python (scripts, load tests), Go (operators, K8s controllers), and YAML (unfortunately). CUDA is worth being able to *read* — not write, just understand why a kernel fusion matters and what a grid/block size means. Rust fluency is a genuine differentiator if you want to contribute upstream.

---

## 17. What CPU vs. Memory Intensive Means, Summarized

After all of the above, here's the clean summary of which hardware resource each step stresses:

| Step | Hardware | Why |
|---|---|---|
| Tokenization | CPU | Text processing, hash lookups, BPE merges |
| Embedding lookup | GPU memory | Row lookup in large weight matrix |
| Positional encoding | GPU compute | Fast arithmetic, small matrix |
| Prefill (attention + FFN) | GPU compute | Matrix multiplications, all tokens in parallel |
| Decode attention | GPU memory bandwidth | KV cache read per step, memory-bound |
| Decode FFN | GPU compute | Weight matrix multiply per step |
| KV cache management | GPU memory | Allocation, paging, eviction |
| Sampling (logit → token) | GPU compute | Softmax + sample, fast |

The pattern: **prefill is compute-bound, decode is memory-bound**. This is the fundamental tension that all inference optimization — P/D disaggregation, speculative decoding, PagedAttention, quantization — is ultimately trying to resolve.

---

## 18. What an SRE Actually Needs to Know

You don't need to write CUDA. You don't need to derive the attention formula. But you do need a mental model that lets you answer these questions in production:

- **Why is TTFT high?** → Prefill bottleneck or queuing. Long prompts? GPU saturated?
- **Why is ITL degrading?** → KV cache pressure. Too many concurrent requests. Memory bandwidth saturating.
- **Why did the GPU OOM?** → KV cache exhausted. Too many long requests, no eviction headroom.
- **Why is throughput low?** → No continuous batching. Poor concurrency config. Batch size too small.
- **Why does prefix caching not help?** → System prompt is changing per-request. Fix the app layer.
- **Which GPU should I buy?** → For inference: memory bandwidth matters more than FLOP count. H100 > A100 for serving not because of compute but because of HBM3 bandwidth.
- **Why is quantization worth it?** → A 4-bit model serves faster (decode is memory-bound, smaller tensors = faster reads) and fits on cheaper hardware. Quality loss is usually acceptable.

The mental model in one sentence: **LLM inference is split between a compute-hungry prefill phase and a memory-hungry decode phase, connected by a KV cache that is your most critical resource to manage.**

Everything else — PagedAttention, continuous batching, P/D disaggregation, speculative decoding, quantization — is an optimization layered on top of that fundamental structure.

---

## Quick Recap

We covered a lot of ground. Here's the map of where we went:

1. **Inference** = running frozen weights against new inputs. Not training. The hard part is serving it at scale.
2. **Tokenization** = text → token IDs. CPU-bound, fast. BPE vocabulary, ~32K–128K tokens.
3. **Embedding** = token IDs → float vectors. Matrix lookup in the model weight file.
4. **Positional encoding** = tell the model where each token sits. RoPE in modern models.
5. **Prefill** = process full prompt through N transformer layers. Compute-bound. Produces KV cache.
6. **Decode** = generate tokens one at a time. Memory-bound. Each step reads the full KV cache.
7. **Sampling** = logits → probability → token. Temperature, top-k, top-p tune this.
8. **KV cache** = stores K and V tensors to avoid recomputation. Grows per token, finite, precious.
9. **Attention** = every token attends to every other. Dot product of Q against K, weighted sum of V.
10. **PagedAttention** = virtual memory for KV cache. Non-contiguous blocks, shared prefix pages.
11. **GGUF/weights** = the big file = all the weight matrices the model uses for matrix math.
12. **Quantization** = compress weights to lower precision. Smaller file, faster decode, slight quality cost.
13. **Continuous batching** = batch at iteration level, not request level. Keeps GPU busy.
14. **Metrics** = TTFT (prefill + queue), ITL/TPOT (decode speed), KV cache usage (capacity).
15. **Languages** = Python (logic), CUDA (kernels), Rust (fast utils), Go (operators).


---

## Summary: The TL;DR

LLM inference isn't magic; it's a measurable, optimizable systems engineering challenge. The execution pipeline breaks down into three distinct phases with unique bottlenecks: Tokenization (CPU-bound), Prefill (GPU compute-bound), and Decode (GPU memory-bound). The ultimate constraint in scaling model serving isn't raw compute power, but memory bandwidth and the strict management of the KV Cache. By understanding the mechanical realities beneath the hood—like PagedAttention, continuous batching, and quantization—infrastructure engineers can move past guesswork and scientifically optimize for the metrics that dictate user experience: TTFT and TPOT.

---

## Conclusion: The Field Is Moving, The Fundamentals Aren't

The implementation details of LLM inference in 2025 are changing fast enough to give you whiplash, but the underlying physics of the problem haven't moved an inch. Prefill will always be compute-hungry. Decode will always be memory-hungry. Attention will always scale quadratically with context length unless someone breaks the math. These aren't framework quirks; they are the mechanical realities of how transformers work.

Transitioning to LLMOps requires a fundamental shift in how we manage system state. We are no longer scaling stateless pods; we are actively managing distributed GPU memory. The engineering headroom to optimize this is enormous, and the landscape is shifting rapidly:

  - **The model size curve is bending:** Smaller, highly-optimized 7B models are now punching above the weight of older 70B giants, distributing the inference problem across a much wider variety of hardware.

  - **The memory bottleneck is softening:** Bleeding-edge KV cache compression techniques are reducing per-token memory footprint from kilobytes down to a handful of bytes, loosening the strict constraints of the decode phase.

  - **The edge is becoming viable:** As inference pushes to mobile NPUs and WebGPU, serving a 3B parameter model starts to look less like a Kubernetes workload and more like a firmware binary.

Yet, the operational reality remains unchanged. When a model is in production and serving real traffic, someone has to know why TTFT spiked at 2am, why the KV cache hit 95% utilization under Tuesday's load, and why the p99 ITL is three times the p50.

Mastering the mechanics of inference separates a fragile AI prototype from a resilient production platform. The inference stack will keep evolving. The need for a rigorous mental model to debug it won't.

---

**Next up:** In the next post, we’ll take these first principles and see how they dictate the architectural trade-offs behind the major inference engines: Ollama, vLLM, TGI, and TensorRT-LLM.


---

*I'm an infrastructure engineer with 11+ years in distributed systems (D-Wave, Enbala, MasterCard, Cisco), currently going deep on LLM serving and inference optimization. This series is grounded in hands-on experiments — Mac Mini to Lambda Labs GH200 to RunPod A100 clusters. I write what I actually learned, including the parts that didn't work.*

*Find me on GitHub: [kraghavan](https://github.com/kraghavan)*

*Find me on Linkedin: [Karthika Raghavan](https://linkedin.com/in/karthikaraghavan)*