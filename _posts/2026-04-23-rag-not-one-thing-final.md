---
layout: single
title: "RAG Is Not One Thing — An Engineer's Guide to Retrieval-Augmented Generation"
date: 2026-04-21
categories: [llm-systems, architecture]
tags: [rag, vector-database, graph-rag, llm, retrieval, embeddings, bm25, agentic-rag, knowledge-graph, production-ai]
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: >
  Most teams implement RAG by reaching for a vector database. That works —
  until it doesn't. An engineer's guide to the full RAG landscape:
  vector, graph, vectorless, hybrid, self-correcting, agentic, and cache-augmented
  retrieval — with a decision framework for choosing the right one.
---

Most teams discover RAG the same way: the LLM doesn't know something recent, doesn't know something internal, or confidently makes something up. Someone reads a blog post about vector databases, spins up Pinecone or pgvector, embeds their documents, and declares the problem solved.

Sometimes it is solved. Often it isn't — and the failure mode is subtle. Retrieval returns chunks that are semantically adjacent but contextually wrong. The LLM gets plausible-sounding context and produces a confident wrong answer. The system works in demos and breaks in production on exactly the queries that matter most.

The problem is usually not the LLM. It's the retrieval architecture — and specifically, the assumption that one retrieval pattern fits every knowledge problem.

This post maps the full RAG landscape. Not just "here are the types" — but **where each pattern belongs in a system, why it wins in some scenarios and fails in others, and how to choose the one your problem actually needs.**

---

## The Mental Model: RAG as Externalised Memory

Before the taxonomy, one framing worth internalising:

**RAG is a pattern for externalising memory from the model's weights into a queryable store.**

A language model's weights contain everything it learned during training — compressed, lossy, and frozen at a point in time. RAG is the architectural response to three limitations of that frozen memory:

- **Staleness** — training cutoffs mean the model doesn't know what happened last month
- **Privacy** — your internal documents were never in the training data
- **Precision** — weights store approximate statistical patterns, not exact facts

The retrieval mechanism is the bridge between a user query and the external knowledge store. What that bridge indexes, how it searches, and what it returns — determines everything downstream. The LLM is not the hard part. **The retrieval layer is the hard part.**

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/rag-externalized-memory.png" alt="Retrival Augumented Memory - Externalized Memory" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
    Architectural overview of Retrieval-Augmented Generation (RAG). By decoupling the stateless inference engine (frozen LLM weights) from the dynamic knowledge layer (Documents, Databases, Knowledge Graphs), RAG enables real-time memory access without the need for model retraining.
  </figcaption>
</figure>

---

## The Foundation: How Words Become Vectors

Every RAG pattern in this post — except Cache-Augmented — depends on a shared primitive: the embedding. Before the taxonomy, it is worth understanding what an embedding actually is, how it's created, and what the geometry means. This is the math that every RAG system is built on top of.

### Step 1: Tokenisation — Words to Integer IDs

The first thing that happens to text in any modern ML system is tokenisation. Words are not the atomic unit — subword tokens are. Most models use Byte Pair Encoding (BPE): a vocabulary of roughly 30,000–100,000 subword pieces is learned from a training corpus, and any text is decomposed into that vocabulary.

```
"retrieval-augmented generation" 
  → ["ret", "rieval", "-", "aug", "mented", " generation"]
  → [4311, 17842, 12, 7816, 972, 9282]   ← integer token IDs
```

These integers have no geometry. `4311` is not "close" to `4312`. They are arbitrary indices into a lookup table — the embedding matrix.

### Step 2: The Embedding Matrix — IDs to Vectors

The embedding matrix `E` has dimensions `[vocab_size × d_model]`. Each row is a learned vector for one token. To embed a token, you do a lookup: retrieve row `i` from `E` for token ID `i`.

```
token ID 4311  →  E[4311]  =  [0.21, -0.84, 0.13, ..., 0.67]  ← 768 numbers
```

This vector — 768 or 1536 or 3072 floating-point numbers — is the embedding. It is a point in high-dimensional space.

**Where do these numbers come from?** They are learned during training via backpropagation, shaped by the training objective:

- **General LLMs** (GPT-4, Claude, Llama): trained on next-token prediction. The embedding space is organised around predictive context — tokens that appear in similar contexts get similar embeddings. "bank" (financial) and "bank" (river) end up in different regions because their context distributions differ.

- **Dedicated embedding models** (`bge`, `e5`, `text-embedding-3`, `Cohere embed-v3`): trained with **contrastive loss**. Semantically similar sentence pairs are pulled together in the space; dissimilar pairs are pushed apart. This produces embeddings far better suited for retrieval than a general LLM encoder — the space is explicitly shaped for similarity search, not next-token prediction.

This distinction matters in practice: using a general LLM's hidden states for RAG embeddings is a common mistake. Use a dedicated embedding model.

### Step 3: The Geometry — Cosine Similarity

Given two vectors `A` and `B`, their similarity is measured by the **cosine of the angle between them**:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

The result is a scalar between -1 and 1:
- **1.0** — identical direction, maximally similar
- **0.0** — orthogonal, unrelated
- **-1.0** — opposite direction, maximally dissimilar

In practice, well-trained embedding models produce retrieval scores in a compressed range — often 0.70–0.90 for everything that's "in the domain." This compression matters, and it's the root of a real architectural problem we'll return to.

**Why cosine over Euclidean distance?** Cosine similarity is invariant to vector magnitude — it measures the *angle*, not the *length*. Embedding norms vary by token frequency and context, so magnitude is noise; angle captures semantic direction. This is why every major vector database defaults to cosine or inner product search.

### The Key Implication for RAG

The geometry of embedding space reflects distributional patterns from training data. **"Similar" in embedding space means "appeared in similar contexts during training"** — not necessarily "answers the same question."

A chunk about Python garbage collection and a chunk about Python memory management will be geometrically close. Whether either one answers your specific query depends on semantics the geometry can only approximate. This is not a bug in embedding models — it is a fundamental property of the representation. It explains every failure mode in Vector RAG, and it's the reason every other pattern in this post exists.

### What's Universal vs. What's Vector-Specific

One distinction worth making explicit before the taxonomy:

**The embedding mechanism described above — words to token IDs to vectors — is universal.** Every pattern in this post involves an LLM at some point, and every LLM operates on embeddings internally. When a SQL RAG system generates a SQL query, when a Graph RAG pipeline extracts entities from a document, when Self-RAG emits a reflection token — the model is running words through an embedding matrix at every step. In that sense, the Foundation section applies to all of them.

**The cosine similarity geometry is not universal.** It is the *retrieval mechanism* for Vector RAG and the dense half of Hybrid RAG. The other patterns replace cosine search with a different organising principle entirely:

- **BM25** — term frequency and inverse document frequency; pure lexical math, no vectors in the retrieval step
- **Graph RAG** — graph traversal along entity edges; retrieval is a walk on a network, not a similarity search
- **SQL RAG** — query execution against a relational schema; the answer is computed, not retrieved by proximity
- **Hierarchical / Structural** — document structure navigation; heading hierarchies and section indices, no embedding lookup
- **Cache-Augmented** — no external retrieval at all; the LLM's own attention mechanism does the work across a context window that already contains the corpus

The patterns diverge at the retrieval layer. They converge again at the LLM layer, where everything is embeddings once more.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/modern-embedding-pipeline.png" alt="Modern Embedding Pipeline" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  The top stage, labeled "Stage 1: Tokenization," takes the input text phrase "retrieval-augmented generation" and visualizes it being processed. Following a downward arrow, the "Stage 2: Embedding Matrix Lookup" box illustrates the use of token IDs to access pre-trained representations. The final stage, labeled "Stage 3: Vector Space," provides a visual model of how these embeddings cluster semantically.
  </figcaption>
</figure>

---

## The Full RAG Taxonomy

Seven meaningfully distinct RAG patterns. They differ in **how they store knowledge, how they search it, and what kinds of queries they handle well.**

| Pattern | Storage | Retrieval Mechanism | Best For |
|---|---|---|---|
| Vector RAG | Vector database | Semantic similarity (ANN) | Unstructured, multi-doc corpora |
| Graph RAG | Knowledge graph | Graph traversal | Entity-heavy, multi-hop queries |
| Vectorless RAG | Inverted index / SQL / structured index | Keyword match / SQL / navigation | Structured docs, tabular data |
| Hybrid RAG | Vector DB + inverted index | Dense + sparse fusion (RRF) | Mixed corpora, production default |
| Self-RAG / Corrective RAG | Any of the above | Conditional + self-evaluated retrieval | High-stakes, hallucination-sensitive |
| Agentic RAG | Any of the above | LLM-directed, multi-step | Complex, multi-hop questions |
| Cache-Augmented | LLM KV cache (in-context) | None — corpus lives in context window | Small, stable, high-query-rate corpora |

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/rag-patterns.png" alt="Retrival Augumented Memory - Externalized Memory" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  A visual guide to seven essential Retrieval-Augmented Generation (RAG) architectures, including Vector, Graph, Hybrid, and Agentic models. Each branch defines the core mechanism and primary use cases using clean diagrams and neon technical styling. 
  </figcaption>
</figure>

---

## 1. Vector RAG — Semantic Similarity Search

Vector RAG is the pattern most engineers reach for first. For unstructured document search it is genuinely the right starting point — just not the only tool.

### How It Works

Two phases: ingestion (offline) and query (online).

**Ingestion pipeline:**
```
Documents → Chunking → Embedding model → Vector database
```

Documents are split into segments (chunks), each chunk is converted into a dense vector by an embedding model, and those vectors are stored. The embedding is a mathematical representation of the chunk's meaning — similar meaning maps to nearby points in high-dimensional space.

**Query pipeline:**
```
User query → Embedding model → ANN search → Top-K chunks → LLM context → Response
```

At query time, the question is embedded with the same model, and the database finds the stored vectors closest to it — approximate nearest neighbours (ANN). Those chunks are assembled into the LLM's context window.

![Vector RAG pipeline — data embedding, semantic space mapping, query routing, neighbourhood-scoped retrieval, and LLM enrichment](/assets/images/rag-as-memory/vector-RAG.png)
*Five-stage vector RAG pipeline: from document ingestion through to LLM semantic enrichment.*

### Chunking: The Most Underrated Decision

Chunking strategy affects retrieval quality more than most teams realise. There is no universally correct chunk size — it depends on your documents and your queries.

**Fixed-size chunking** (e.g., 512 tokens with 50-token overlap) is the default because it's simple. It's also wrong for most document types. A paragraph boundary mid-sentence, a heading detached from its body, a table split in half — all produce chunks that are retrievable but semantically broken.

**Semantic chunking** splits at natural boundaries: paragraph breaks, section headings, topic shifts. More expensive to compute at ingestion time, materially better precision at query time.

**Hierarchical chunking** maintains two granularities simultaneously — parent section and child paragraph. Retrieval finds the child; context delivery includes the parent for surrounding meaning. LlamaIndex calls this "small-to-big retrieval." Worth the implementation overhead for long documents.

**Late chunking** is the most recent approach: embed the full document, then produce chunk-level embeddings that retain document-wide context. Requires models that support this natively — JinaAI's `jina-embeddings-v3` is the current reference implementation.

> **Author's note:** If your RAG system performs poorly, check chunking before you blame the embedding model or the LLM. Chunking errors are silent — the system retrieves *something*, just not the right thing.

### Embedding Models: The Consequential Choice

The embedding model encodes what "similar" means. It's the most consequential decision in vector RAG.

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 (reducible) | Strong general-purpose, managed API |
| `embed-v3` (Cohere) | 1024 | Excellent multilingual, query/doc prefix support |
| `bge-large-en-v1.5` (BAAI) | 1024 | Best open-source general benchmark |
| `e5-mistral-7b-instruct` | 4096 | LLM-based, expensive but strong on technical domains |
| `jina-embeddings-v3` | 1024 | Late chunking support, task-specific LoRA |

**Critical constraint:** The embedding model must be **identical** for ingestion and query. Embedding documents with one model and queries with another means searching in the wrong space. Changing models mid-production requires re-embedding the entire corpus.

**Domain fit matters.** Models trained on general web text may not produce useful embeddings for highly technical, legal, or medical corpora. Evaluate on your domain, not just on general benchmarks like MTEB.

### Vector Databases: The Operational Choice

The vector database is mostly an operational decision. They all do ANN search — they differ in ops complexity, filtering capabilities, and scale.

| Database | Deployment | Notes |
|---|---|---|
| FAISS | In-memory, no persistence | Local dev, research |
| pgvector | Postgres extension | Simple ops for teams already on Postgres |
| Pinecone | Fully managed | Production-ready, no infra overhead |
| Weaviate | Self-hosted or managed | Schema-aware, native hybrid search |
| Qdrant | Self-hosted | High throughput, strong metadata filtering |
| Chroma | Local / embedded | Prototyping only |

Metadata filtering — "find relevant chunks but only from documents published after Q3 2024, authored by this team" — requires your database to support it. Not all do equally well, and filtered ANN search has varying performance characteristics across implementations.

### When Vector RAG Wins

Strong when the query is **semantically open-ended** and the corpus is **unstructured and variable**.

- Customer support over a large unstructured knowledge base
- Research assistant over a corpus of PDFs and papers
- Internal wiki search where the same concept is described differently across different documents
- Semantic recommendation ("find content similar to this")

### When Vector RAG Fails

**Irrelevant matches at high similarity.** Two chunks can score closely — same vocabulary, same domain — but answer completely different questions. A chunk about "Python memory management" and one about "Python garbage collection" score similarly for many queries. The LLM receives the wrong one and generates a plausible answer from the wrong source. You won't know unless retrieval is instrumented separately from generation.

**Chunking destroys long-document coherence.** A document's conclusion may reference its introduction. Retrieving the conclusion without the introduction gives the LLM half a picture.

**Exact fact retrieval.** "What was the Q3 revenue figure?" requires exact match, not semantic proximity. BM25 or SQL is the right tool.

**Long structured documents.** Legal contracts, technical specifications, regulatory filings — these have rigid internal structure. The answer to "what is the termination clause?" is the specific section with that heading, not the chunk most similar to the word "termination."

### The Scale Problem: When Retrieval Loses Its Discriminative Power

There is a mathematically grounded reason why flat vector RAG degrades as your corpus grows — and a highly publicised but largely fabricated version of that claim that swept through tech communities in early 2026.

**The real finding.** Stanford researchers (Dahl et al., *Journal of Empirical Legal Studies*) evaluated commercial enterprise RAG tools built by legal giants — Lexis+ AI, Westlaw AI-Assisted Research — against structured legal corpora. Despite state-of-the-art retrieval, the systems hallucinated **17–33% of the time**. The failure wasn't a mathematical breakdown of the vector database. It was something subtler: as corpus size grows, naive RAG increasingly retrieves documents that are *semantically similar but contextually wrong* — an outdated ruling, a correctly matched case from the wrong jurisdiction. The LLM receives plausible-looking context and synthesises a confident, wrong answer.

**The viral myth built on top of it.** In January 2026, a wave of influencer posts (circulating as *"Stanford's Warning: Your RAG System Is Broken"*) took the Dahl et al. findings and layered fabricated statistics on top: an 87% precision drop past 50,000 documents, AI "getting dementia" at exactly 10,000 documents because embeddings become entirely equidistant. The AI engineering community rapidly debunked these specific claims — production systems routinely query millions of vectors successfully. The 10,000-document limit was pure internet myth.

**The legitimate math underneath.** The influencers were borrowing, imprecisely, from a real geometric phenomenon: the **curse of dimensionality**. Here is what actually happens as you scale a high-dimensional vector corpus:

In `d`-dimensional space, volume grows exponentially with `d`. As the number of vectors increases, they concentrate in a thin shell near the outer surface of a hypersphere — and critically, the ratio of maximum to minimum distance between any two points begins converging toward 1. Formally:

```
lim(d→∞) [ (max_distance - min_distance) / min_distance ] → 0
```

In practical terms: cosine similarity scores for all retrieved chunks compress into a narrow band. Where a well-tuned small corpus might return scores spanning 0.60–0.95 (making top-K selection meaningful), a large corpus might return scores spanning 0.78–0.84 across everything. The gap between a highly relevant chunk and a marginally relevant one shrinks — and ranking becomes unreliable.

This is not a cliff at 10,000 documents. It is a gradual degradation that becomes material at enterprise scale — tens of millions of chunks, corpora spanning multiple domains.

**The architectural response** to this is exactly what the rest of this post describes: hybrid search (BM25 lexical retrieval is immune to this geometry problem, operating on term overlap rather than vector angles), cross-encoder reranking (which evaluates the full query-passage pair jointly rather than comparing vectors in isolation), and Graph RAG (which bypasses cosine similarity entirely in favour of graph traversal).

> **Author's note:** If someone cites the "10,000 document limit" to you, they've read the influencer posts, not the paper. The real concern — semantic similarity without contextual correctness, and cosine score compression at scale — is legitimate and worth designing against. The solution is not to abandon vector RAG; it's to layer hybrid retrieval and reranking on top of it.

---

## 2. Graph RAG — Knowledge as a Network

Graph RAG is the most architecturally distinct pattern. Instead of indexing document chunks, it extracts **entities and relationships** from documents and stores them as a knowledge graph. Retrieval is graph traversal, not similarity search.

This is the right pattern when the relationships *between* things matter as much as the things themselves.

### How It Works

**Ingestion — building the graph:**

Documents are processed — by an LLM or an NLP pipeline — to extract two things:
1. **Entities** — people, organisations, products, concepts, locations
2. **Relationships** — labeled edges between entities: "works at", "manufactures", "depends on", "contradicts"

These are stored in a graph database. Neo4j is the production standard. For smaller corpora or research, an in-memory graph (NetworkX, or a custom adjacency structure) is sufficient.

```
Documents → Entity extraction (LLM pass) → Relationship extraction → Graph DB
              ↓
         Entities: [NVIDIA, GH200, NVLink-C2C, Hopper GPU, Grace CPU]
         Edges:    GH200 --[has-component]--> Hopper GPU
                   GH200 --[has-component]--> Grace CPU
                   Hopper --[connects-via]--> NVLink-C2C
```

**Query — traversing the graph:**

The user's question is parsed to identify entities. The graph is traversed from those seed entities — following edges to related nodes, collecting context along the way. The resulting subgraph is serialised to text and passed to the LLM.

```
Query: "How does the GH200 connect its CPU and GPU?"
  → Entity recognition: [GH200, CPU, GPU]
  → Seed node: GH200
  → Traversal: GH200 → Hopper GPU, Grace CPU → NVLink-C2C (via "connects-via" edge)
  → Subgraph serialised to text → LLM → Answer
```

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/graphRag-pipeline.png" alt="Graph RAG Pipeline" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  Comprehensive Graph RAG pipeline diagram: Ingestion processes PDFs into a structured Knowledge Graph (Neo4j) (left); Query processes use Breadth-First Search traversal to extract relevant subgraph context, enabling an LLM to generate precise answers (right).
  </figcaption>
</figure>

### Building a Graph from Markdown

Markdown is an underappreciated source format for Graph RAG. When structure is consistent — heading hierarchy, stable entity naming, internal links — entity extraction becomes significantly more reliable and cheaper.

**Headings become entity types.** An H1 is a primary entity; H2s beneath it are sub-entities or attributes. This hierarchy maps directly to node types in the graph.

**Internal links become edges.** `[GH200](./gh200.md)` is a relationship you get for free. The link anchor text is often a usable relationship label if your team names links carefully.

**Frontmatter becomes metadata.** `tags:`, `category:`, `related:` fields are pre-extracted attributes — no LLM pass required.

**Consistent naming is load-bearing.** If your markdown uses "NVIDIA GH200", "GH200 GPU", and "Grace Hopper" interchangeably for the same thing, the graph fragments into disconnected islands. Entity resolution — recognising these as the same node — is a non-trivial engineering problem. Invest in naming conventions *before* you invest in Graph RAG infrastructure.

A practical extraction pipeline for a markdown knowledge base:

```python
# Pseudo-code — actual implementation uses spaCy or an LLM extraction prompt
for file in markdown_files:
    entities  = extract_entities(file.content)    # NER or LLM
    relations = extract_relations(file.content)   # relation extraction
    metadata  = parse_frontmatter(file)           # free, no ML required
    links     = parse_wikilinks(file)             # free, structural
    graph.add(entities, relations, metadata, links)
```

### Microsoft's GraphRAG: Community Detection

Microsoft Research published GraphRAG in 2024 with an important extension: **community detection**. After building the entity graph, a clustering algorithm (Leiden, by default) groups related entities into communities — thematic clusters of connected nodes. Each community gets an LLM-generated summary.

This enables two levels of retrieval:

- **Local search** — specific entity questions: traverse the graph from seed entities, return the subgraph
- **Global search** — corpus-wide synthesis questions: query community summaries, then drill into relevant communities

The key insight from the paper: standard RAG fundamentally fails at **global questions** that require synthesising information across an entire corpus. "What are the major risk themes across all these financial filings?" cannot be answered by retrieving top-K chunks — you'd need all the documents. Community summaries compress the corpus into a navigable hierarchy without loading everything into the context window.

### When Graph RAG Wins

**Entity-heavy domains.** Medical records (patient → diagnosis → treatment → drug → side-effect), legal documents (contract → party → clause → obligation → penalty), software dependency graphs (service → depends-on → service → owned-by → team). Any domain where the answer to a question lives in the *path* between entities, not inside a single document chunk.

**Multi-hop reasoning.** "Which customers have purchased products manufactured by suppliers that had quality incidents in the last six months?" This traverses: customer → purchase → product → manufacturer → quality-event → date. Vector search over chunked documents cannot answer this reliably. A graph walk can.

**Global synthesis.** "What are the recurring themes across all our internal architecture RFCs?" This is the Microsoft GraphRAG community-summary use case.

**Structured markdown knowledge bases.** Obsidian vaults, wiki systems, and engineering documentation repos with consistent heading hierarchies and internal linking are excellent Graph RAG candidates.

### When Graph RAG Fails

**It is expensive to build and maintain.** Entity extraction requires an LLM pass over every document at ingestion time — budget for this compute cost. Graph updates when documents change require reconciling entity and edge changes, not just re-embedding a chunk.

**Query latency is unpredictable.** Simple traversals are fast. Deep traversals on large, dense graphs can be slow in ways that are hard to bound without profiling.

**Entity resolution is harder than it looks.** Synonyms, abbreviations, and capitalisation variants all create phantom duplicate nodes that fragment traversal results. This is not a solved problem.

> **Author's note:** Graph RAG is frequently over-applied. The majority of production RAG use cases need precise chunk retrieval, not relationship traversal. Build Graph RAG when your domain has a clear entity-relationship structure *and* users demonstrably ask multi-hop questions. Validate the need first — the infrastructure commitment is significant.

---

## 3. Vectorless RAG — Structural Navigation

Vectorless RAG is the underappreciated half of the retrieval landscape. No embeddings, no vector similarity — it navigates documents by structure, keywords, or schema. The tagline from a graphic that's been circulating in the ML community puts it well:

> **Vector RAG finds similar text. Vectorless RAG finds the right place.**

![Full RAG end-to-end architecture — ingestion pipeline (offline) feeding a 
knowledge store, query pipeline (online) from user query through rewriting, 
retrieval, optional evaluation, reranking, context assembly, and generation, 
with an evaluation feedback loop](/assets/images/rag-as-memory/vector-vectorless-rag-2.png)
*Full RAG system architecture. Entity extraction in the ingestion pipeline applies 
to Graph RAG specifically — Vector and Vectorless RAG skip that step and go directly 
from chunking to their respective stores.*
*The risk profiles are mirror images: vector RAG risks irrelevant matches at high similarity; vectorless RAG risks failing when structure is inconsistent.*

### BM25 / Sparse Retrieval

The classic information retrieval algorithm. Documents are indexed as term-frequency vectors. Queries match documents based on keyword overlap, weighted by term rarity (IDF — rare terms that appear in your query and your document are strong signal). BM25 is what Elasticsearch and OpenSearch use under the hood.

**Wins when:** The query contains specific terminology that must appear in the result — product codes, legal terms, named entities, exact model numbers. It's also highly interpretable: you can explain exactly why a document was returned, which matters in auditable systems.

**Fails when:** The document says "contract termination" and the query says "ending the agreement." BM25 returns nothing. Paraphrase is its fundamental blind spot.

### SQL RAG (Text-to-SQL)

The knowledge lives in a relational database. The LLM generates a SQL query from the user's natural language question, executes it against the database, and uses the structured result as context.

```
User: "What was total revenue by region in Q3?"
  → LLM generates: SELECT region, SUM(revenue) FROM sales
                   WHERE quarter = 'Q3' GROUP BY region
  → Execute against DB → Structured result → LLM formats → Response
```

Precision is exact — no approximation, no semantic drift. The risks are SQL generation errors on complex schemas and LLM hallucination of column or table names. Mitigations: include schema documentation in the system prompt, validate SQL before execution, enforce read-only access.

**Wins when:** The knowledge is inherently tabular — sales data, inventory, user records, metrics. Any time the answer requires aggregation, filtering, or joining across structured records.

### Hierarchical / Document-Structure Navigation

Documents are indexed by their structural elements: headings, sections, metadata. Query routing uses the document's own organisation rather than semantic proximity.

"Find the termination clause" navigates to the section with that heading — not the chunk most similar to the word "termination" spread across a thousand chunks. For well-structured documents, the precision advantage over vector retrieval is substantial.

**Wins when:** Documents have reliable, consistent structure — contracts, API documentation, regulatory filings, technical specifications. Chunking these documents would split answers across chunk boundaries and destroy retrieval precision.

---

## 4. Hybrid RAG — Dense + Sparse Combined

Hybrid RAG runs vector similarity search and BM25 sparse retrieval in parallel, then merges the results. The standard fusion approach is **Reciprocal Rank Fusion (RRF)** — rank candidates from each retriever independently, then combine ranks to produce a unified list.

```
Query → Dense retriever  → ranked list A ─┐
      → Sparse retriever → ranked list B ─┤─→ RRF fusion → unified list → reranker → LLM
```

**Why it works:** Dense retrieval is strong on semantics and paraphrase; sparse retrieval is strong on keywords and exact terms. A query like "what is NVIDIA's NVLink-C2C bandwidth specification?" benefits from both — semantic understanding of the concept *and* exact keyword matching of the specification figure.

**Reranking** is a common addition: a cross-encoder model that scores (query, chunk) pairs more precisely than initial retrieval. Cohere Rerank, BGE Reranker, and Jina Reranker are the standard choices. Reranking is expensive (linear in retrieved candidates) but materially improves precision for high-value queries. Add it when precision matters more than the latency cost.

**When to default to this:** Hybrid RAG is the sensible production default for mixed corpora — some documents structured, some not, queries varying in specificity. Weaviate has native hybrid search. For Pinecone or pgvector, you run both retrievers and implement RRF yourself, which is roughly 30 lines of Python.

---

## 5. Self-RAG / Corrective RAG — Retrieval That Checks Its Own Work

Standard RAG retrieves once and trusts the result. The pipeline has no feedback: whatever the retriever returns becomes the LLM's context, regardless of whether it actually answers the question. Self-RAG and Corrective RAG both address this — through different mechanisms.

### The Problem They Solve

Consider a question about a niche regulatory change. Standard RAG retrieves the top-K chunks by similarity, hands them to the LLM, and the LLM generates an answer — even if the retrieved chunks are about the wrong regulation, from the wrong year. The output looks confident. The user has no signal that something went wrong.

In high-stakes domains — medical, legal, financial — a confident wrong answer is worse than no answer.

### How Self-RAG Works

The Self-RAG paper (Asai et al., 2023) proposes training the model to emit **reflection tokens** inline with its output — metacognitive signals that critique the retrieval and generation process as it runs.

Four token types:

- **[Retrieve]** — should I retrieve at all for this query, or can I answer from weights?
- **[IsRel]** — is this retrieved passage actually relevant to what was asked?
- **[IsSup]** — does my generated output follow from the retrieved passage, or am I confabulating?
- **[IsUse]** — is my overall response useful and grounded?

This makes retrieval **conditional and self-critical**:

```
Query arrives
  → [Retrieve: Yes/No] — skip retrieval if the model is confident without it
  → Retrieve (if Yes)
  → [IsRel: Relevant/Irrelevant] — score each retrieved passage
  → Generate (using only relevant passages)
  → [IsSup: Fully/Partially/Not] — score whether the generation is grounded
  → [IsUse: score] — evaluate overall response utility
  → Return if grounded, else re-retrieve with refined query
```

The practical upside: the model becomes its own quality gate. It doesn't blindly trust retrieval, and it doesn't generate past what its context can support.

### How Corrective RAG (CRAG) Works

CRAG takes a related but architecturally distinct approach. Instead of the generation model critiquing its own output, a **separate lightweight evaluator model** scores retrieval quality before generation begins. If the score falls below a threshold, the system doesn't proceed to generation — it corrects first.

```
Query → Retrieve
          ↓
     Evaluator model scores retrieved context
          ↓ Score: High        ↓ Score: Low / Ambiguous
       Generate          Reformulate query or fall back to web search
                                  ↓
                             Retrieve again → Evaluate → Generate
```

The correction strategies:
- **Query reformulation** — rewrite the question and re-retrieve from the same store
- **Web search fallback** — if internal retrieval fails, escalate to a broader search
- **Knowledge refinement** — strip irrelevant passages from the retrieved set before passing to the LLM

**Self-RAG vs CRAG in practice:** Self-RAG is baked into the model — it requires a fine-tuned model that knows about reflection tokens. CRAG is a pipeline component — a separate evaluator you attach to any existing RAG system. CRAG is significantly easier to adopt without retraining.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/self-rag-crag-compare.png" alt="Comparing Self-Rag and CRAG" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  This comparison highlights the fundamental difference between Self-RAG and CRAG. Self-RAG integrates internal reflection tokens directly into the generation process to guide its own retrieval and output refinement, while CRAG utilizes a separate evaluator model within a modular pipeline to correctively refine knowledge retrieval.
  </figcaption>
</figure>

### When Self-RAG / Corrective RAG Wins

**High-stakes QA where hallucination is unacceptable.** Medical diagnosis support, legal research, financial analysis. Any domain where user trust is eroded by a single confident wrong answer.

**Variable retrieval quality.** If your corpus is inconsistent — some documents well-structured, some noisy — retrieval quality varies significantly by query. Self-RAG and CRAG both provide a safety net that standard RAG lacks.

**Queries where retrieval might not be needed at all.** Self-RAG's `[Retrieve: No]` token lets the model skip retrieval for questions it can answer from its weights — reducing latency and cost for queries where the knowledge base adds nothing.

### When Self-RAG / Corrective RAG Fails

**Latency-sensitive applications.** Both patterns add LLM calls to the pipeline — reflection tokens during generation (Self-RAG) or a separate evaluation pass (CRAG). At 2–4 additional LLM calls per query, this is not suitable for sub-500ms response requirements.

**Consistently high retrieval quality.** If your retrieval is already precise — small, well-structured corpus, specific queries, BM25 or SQL RAG — the evaluation overhead adds cost without meaningful quality improvement.

**CRAG's evaluator can be the bottleneck.** The evaluator model needs to be good enough to catch bad retrievals but fast enough not to dominate latency. Calibrating this threshold is non-trivial and requires an evaluation dataset to tune against.

> **Author's note:** Self-RAG and Corrective RAG are evaluation wrappers around your existing retrieval pattern — they don't replace the need for good retrieval. A well-tuned hybrid RAG system with a reranker will outperform Corrective RAG over a poorly chunked, poorly indexed corpus. Fix the retrieval foundation first; add self-correction as a quality gate after.

---

## 6. Agentic RAG — LLM-Directed Retrieval

In standard RAG, the retrieval plan is fixed at design time: embed the query, find similar chunks, pass to LLM. Agentic RAG inverts this: **the LLM decides what to retrieve, when to retrieve it, and whether it has enough context to answer.**

### How It Works

```
User query → LLM agent
               ↓ "I need the termination clause in Contract A first"
           Vector search → Contract A termination clause retrieved
               ↓ "Now I need the equivalent clause in Contract B"
           Vector search → Contract B termination clause retrieved
               ↓ "I need current regulatory context for this clause type"
           Web search → Current regulatory text retrieved
               ↓ "I have sufficient context — synthesising"
           Final answer
```

The LLM acts as a planner: it decomposes the question, issues retrieval calls as tool invocations, evaluates results, decides whether to retrieve further, and ultimately synthesises the final answer. Each retrieval step is driven by what the model has learned so far — not by a fixed pipeline.

**Why this matters:** Complex questions require multi-step information gathering. "Compare the indemnification clauses in these three contracts and flag which one creates the greatest buyer risk" cannot be answered from a single retrieval pass. An agentic system decomposes, retrieves per-contract, and synthesises across all results.

**The tools an agentic RAG system typically exposes to the LLM:**
- Vector search over a document corpus
- SQL query execution against a structured database
- Web search for current or external information
- API calls to internal systems
- Code execution for computation

**The cost:** Each LLM planning call, each retrieval step, each synthesis pass — latency compounds. Agentic RAG runs 5–20x slower and more expensively than standard RAG per query. Use it when query complexity justifies the overhead. Not as a default.

**Frameworks:** LangGraph (recommended for production state management), LlamaIndex Agents, AutoGen. The orchestration layer can wrap any retrieval pattern from this post.

---

## 7. Cache-Augmented Generation — When the Context Window Is the Store

Cache-augmented generation takes a fundamentally different architectural position. There is no external retrieval store, no vector database, no ANN search. Instead, the entire knowledge corpus is loaded directly into the LLM's context window — and the LLM's own attention mechanism does the "retrieval."

### How It Works

When a transformer processes a sequence of tokens, it computes key-value (KV) attention matrices that represent how each token attends to all prior tokens. These matrices can be computed once for a static prefix — the knowledge corpus — and **cached on disk or in GPU memory**. Every subsequent query shares this cached computation; only the query tokens themselves require fresh processing.

```
Step 1 (one-time, expensive):
  Full corpus → LLM forward pass → KV cache computed → Saved to storage

Step 2 (per query, cheap):
  Cached KV prefix + User query → Only query tokens processed → Response
```

Anthropic calls this **prompt caching**. Google's Gemini API has an equivalent under the name **context caching**. The pricing model reflects the asymmetry: the initial cache-fill pass is charged at full input token rates; subsequent queries that reuse the cache are charged at a fraction of that cost (roughly 10% of standard input pricing in current API offerings).

**What "retrieval" looks like here:** There is none in the traditional sense. The LLM's attention mechanism — which already knows how to find relevant parts of its context — does the work. You're not searching for the right chunk; you're trusting the model to attend to the right parts of a context that contains everything.

### When Cache-Augmented Wins

**Small, stable, high-query-rate corpora.** An API documentation assistant where the same 300-page reference manual is queried thousands of times per day. A customer support system over a fixed product handbook that changes quarterly. An internal code assistant over a stable, bounded codebase.

**Eliminating retrieval latency.** Cache-augmented generation has the lowest query-time latency of any RAG pattern — there is no retrieval step. Once the KV cache is warmed, response latency is dominated only by generation, not by vector search or graph traversal.

**Corpora that fit in the context window.** Long-context models make this practical for meaningfully large knowledge bases. Gemini 2.0's 1M-token context accommodates roughly 750,000 words — the equivalent of several large technical handbooks or a substantial regulatory corpus.

**Avoiding embedding model lock-in.** There is no embedding model to choose, maintain, or migrate. The same model that generates answers also "searches" the corpus.

### When Cache-Augmented Fails

**Corpora that exceed the context window.** This is the hard ceiling. No chunking strategy or retrieval trick helps — if the corpus doesn't fit, this pattern doesn't apply.

**Frequently updated corpora.** When documents change, the KV cache is stale. Re-filling the cache means re-processing the full corpus, paying the full input token cost again. For corpora that update daily or more frequently, this cost compounds quickly.

**Cold-start, one-off queries.** The economic model only makes sense if the cache is reused many times. For one-off queries where the corpus is loaded fresh and queried once, cache-augmented is strictly more expensive than retrieval-based alternatives.

**Precision-sensitive applications over large corpora.** Trusting the attention mechanism to find the right passage in a 500K-token context is not the same as targeted retrieval with a reranker. For applications where the exact source of an answer must be precisely attributed, structured retrieval patterns are more reliable.

> **Author's note:** Cache-augmented generation is best understood as an inference-layer optimisation, not a retrieval pattern in the traditional sense. It trades corpus size flexibility (which retrieval-based patterns handle well) for query latency and operational simplicity (no retrieval infrastructure to build or maintain). The right question is not "should I use cache-augmented instead of vector RAG?" — it's "does my corpus fit in the context window, and do I have enough query volume to amortise the cache-fill cost?"

---

## The Decision Framework

Four questions determine which pattern your problem needs.

**1. What is the structure of your data?**

| Data type | Start here |
|---|---|
| Unstructured text, mixed document types | Vector RAG or Hybrid |
| Consistent internal structure (sections, headings) | Vectorless / Hierarchical |
| Relational / tabular | SQL RAG |
| Entity-rich, relationship-heavy | Graph RAG |
| Small, stable corpus fitting in context | Cache-Augmented |

**2. What kind of question is the user asking?**

| Query type | Start here |
|---|---|
| Semantic / open-ended | Vector RAG |
| Exact keyword or specific term | BM25 |
| Numerical / aggregation | SQL RAG |
| Multi-hop / relationship traversal | Graph RAG |
| High-stakes, hallucination-sensitive | Self-RAG / Corrective RAG |
| Complex, multi-step, requires planning | Agentic RAG |

**3. What is your latency budget?**

| Budget | Options |
|---|---|
| < 100ms | BM25, Cache-Augmented (pre-warmed) |
| 100–500ms | Vector RAG (managed vector DB) |
| 500ms–2s | Hybrid RAG with reranking |
| 2s+ acceptable | Graph RAG, Agentic RAG, Self-RAG / Corrective RAG |

**4. What is your team's operational complexity tolerance?**

| Tolerance | Options |
|---|---|
| Low | Vector RAG (managed), BM25 (Elasticsearch), Cache-Augmented |
| Medium | Hybrid RAG, SQL RAG, Corrective RAG |
| High | Graph RAG, Agentic RAG, Self-RAG (requires fine-tuned model) |

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/optimal-RAG-pattern.png" alt="An Optimal RAG Pattern" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  A professional infographic flowchart for selecting the optimal RAG pattern. Starting with data structure and query types, it guides users through core retrieval approaches (Vector, BM25, Graph, SQL) and integrates specialized patterns (Agentic, Self-RAG, Cache-Augmented) for multi-step reasoning, hallucination control, and high performance. Dark background with teal accents.
  </figcaption>
</figure>

---

## Where RAG Fits in the System Architecture

RAG is not a feature. It is a system that touches every layer from data ingestion to LLM response. Most tutorials cover the query pipeline and skip the rest.

<figure style="max-width:720px;margin:2rem auto;text-align:center;">
  <img src="/assets/images/rag-as-memory/full-RAG-architecture.png" alt="Retrival Augumented Memory - Externalized Memory" style="width:100%;">
  <figcaption style="font-size:0.85rem;color:#888;margin-top:0.5rem;">
  Full RAG system architecture diagram
  </figcaption>
</figure>

**The ingestion pipeline — where most quality problems originate (offline):**

- **Chunking strategy** — the most consequential offline decision. Affects every downstream component. Invest here first.
- **Embedding model** — must be consistent between ingestion and query. Changing models requires re-embedding the entire corpus.
- **Metadata extraction** — source, date, section, author, version. Metadata enables filtered retrieval and is often the difference between a useful and useless system in production.
- **Entity extraction** (Graph RAG only) — an LLM pass over every document at ingestion time. Budget for this compute explicitly.

**The query pipeline — latency-sensitive (online):**

- **Query rewriting** — conversational queries are verbose and poorly suited for retrieval. Rewriting to a clean retrieval query improves recall. High return, low implementation cost.
- **Retrieval** — the pattern choice from this post.
- **Evaluation** (Self-RAG / CRAG) — optional quality gate that adds latency but catches bad retrievals.
- **Reranking** — improves precision at the cost of latency. Add when precision matters more than speed.
- **Context assembly** — the order and formatting of retrieved chunks in the prompt affects generation quality. Placement matters more than most teams expect.

**The feedback loop — most often absent:**

Production RAG systems need evaluation instrumentation from day one. Which queries return irrelevant context? Which queries result in hallucinated answers despite retrieved context? Minimum metrics: retrieval precision, retrieval recall, answer faithfulness, answer relevance. Without this loop, every optimisation is a guess.

---

## What Most Teams Get Wrong

**1. Treating retrieval as solved after the demo works.**

Teams instrument LLM response quality and call it monitoring. The retrieval layer — which determines answer quality — is a black box. Measure retrieval separately from generation. If retrieval is wrong, no prompt engineering saves you.

**2. Chunking by token count instead of by meaning.**

Fixed-size chunking is the default because it's simple. A paragraph split mid-sentence, a heading detached from its body, a table broken in half — all produce chunks that retrieve but don't answer. Semantic, hierarchical, or late chunking strategies are worth the evaluation time against your specific document types.

**3. Assuming vector RAG is the universal answer.**

For a significant fraction of production use cases, BM25 outperforms dense retrieval — faster, more interpretable, no embedding infrastructure to maintain, better precision when queries use the same terminology as documents. Vector RAG is the right default for semantically variable, unstructured corpora. Not for everything.

**4. Building Graph RAG or Agentic RAG before validating the need.**

Both are powerful for the right problems. Both are expensive to build, slow to iterate on, and operationally complex to maintain. Validate that your queries actually require multi-hop traversal or multi-step planning before committing to the infrastructure.

**5. Skipping evaluation entirely.**

"Our answers seem okay — how do we know if they're good?" You don't, without instrumentation. Ground-truth query sets, automated faithfulness scoring, retrieval hit rate — build this early. Every optimisation decision depends on it.

---

## Where to Start

**For a new RAG system:** Start with **Hybrid RAG**. Vector retrieval covers semantic queries; BM25 covers keyword queries; RRF fusion is straightforward to implement. Most production corpora benefit from both. Add a reranker if your latency budget allows.

**Add Graph RAG** when your domain has clear entity relationships and users demonstrably ask multi-hop questions. Validate the need with real query examples before building.

**Add Self-RAG / Corrective RAG** when hallucination risk is unacceptable and retrieval quality is variable — medical, legal, financial use cases are the canonical fits. Use Corrective RAG (CRAG) if you can't retrain the model; Self-RAG if you can.

**Add Agentic RAG** when queries require multi-step reasoning that can't be predetermined. Budget for the latency and cost — it's not a drop-in replacement for standard RAG.

**Consider Cache-Augmented** when your corpus fits in the context window, query volume is high, and you want to eliminate retrieval infrastructure entirely. Know the cache invalidation cost before committing.

The retrieval mechanism is not the hardest part of RAG. The hardest part is knowing whether your system is returning the right information. Get evaluation instrumented before you optimise anything else.

---

*Platform engineer with 11+ years in distributed systems, currently building deep expertise in LLM serving infrastructure and retrieval systems. Writing about what I'm learning at [kraghavan.github.io](https://kraghavan.github.io).*

*[GitHub](https://github.com/kraghavan) · [LinkedIn](https://linkedin.com/in/karthikaraghavan)*