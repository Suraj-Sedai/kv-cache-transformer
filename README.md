# MiniGPT Inference Engine with KV Cache

A from-scratch Transformer **inference-focused implementation** that demonstrates how modern Large Language Models perform **efficient autoregressive decoding** using **Key–Value (KV) caching**. This project emphasizes *correctness*, *systems-level understanding*, and *engineering clarity* over raw scale.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [High-Level Architecture](#high-level-architecture)
4. [Transformer Components](#transformer-components)
   - Token & Positional Embeddings
   - Transformer Block
   - Causal Self-Attention
5. [KV Cache Design](#kv-cache-design)
   - What is KV Caching?
   - Why KV Cache Works
   - Per-Layer Cache Isolation
6. [Decoding Paths](#decoding-paths)
   - Full (Non-Cached) Decoding
   - Incremental (Cached) Decoding
7. [Logits Processing & Sampling](#logits-processing--sampling)
   - Temperature
   - Top-k
   - Top-p (Nucleus Sampling)
   - Repetition Penalty
8. [Correctness Validation](#correctness-validation)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
11. [Project Status & Extensions](#project-status--extensions)

---

## Project Overview

This repository implements a **MiniGPT-style Transformer inference engine** with:

- Autoregressive decoding
- Layer-wise KV caching
- Incremental token generation
- Production-style logits processing
- Correctness and performance validation

The goal is to **understand and reproduce the mechanics of real LLM inference**, similar to systems used in GPT, LLaMA, and modern inference frameworks.

---

## Motivation

Naïve autoregressive decoding recomputes attention over the *entire sequence* for every new token, resulting in **O(T²)** complexity per step.

Modern LLMs avoid this using **KV caching**, reducing decoding complexity to **O(T)** per step.

This project answers:
- *How exactly does KV caching work?*
- *Why must caches be per-layer?*
- *How do positional embeddings stay aligned?*
- *Why does cached decoding remain mathematically equivalent?*

---

## High-Level Architecture

```
Input Token
   │
   ▼
Token Embedding + Positional Embedding
   │
   ▼
[ Transformer Block × N ]
   │      └── KV Cache (per layer)
   ▼
LayerNorm
   │
   ▼
LM Head → Logits
   │
   ▼
Logits Processor → Sampling → Next Token
```

---

## Transformer Components

### Token & Positional Embeddings

- Token embeddings map vocabulary IDs to dense vectors.
- Positional embeddings encode token order.

**Key invariant:**
- Full decoding uses positions `[0 … T-1]`
- Cached decoding uses position `t` at step `t`

Incorrect positional alignment breaks KV caching.

---

### Transformer Block

Each block contains:

1. LayerNorm
2. Cached Causal Self-Attention
3. Residual connection
4. Feed-Forward Network

Each block maintains **its own KV cache**, ensuring:
- Independent learned projections
- Correct attention behavior

---

### Causal Self-Attention

- Queries (`Q`) are computed **only for the current token** during cached decoding.
- Keys (`K`) and Values (`V`) from previous steps are reused from cache.

Causal masking prevents attention to future tokens.

---

## KV Cache Design

### What is KV Caching?

KV caching stores:
- Projected Keys and Values
- For every layer
- For all previously generated tokens

This avoids recomputing them during decoding.

---

### Why KV Cache Works

At decoding step `t`:
- Only `Q_t` depends on the new token
- `K_0…t-1` and `V_0…t-1` are unchanged

Thus:
```
Attention(Q_t, [K_0…K_t], [V_0…V_t])
```
can reuse cached values without changing results.

---

### Per-Layer Cache Isolation

Each Transformer layer has:
- Different learned projection matrices
- Different attention distributions

Sharing KV cache across layers is **incorrect**.

---

## Decoding Paths

### Full (Non-Cached) Decoding

- Recomputes attention for all tokens
- Used for training and validation
- Complexity: **O(T²)** per step

---

### Incremental (Cached) Decoding

- Processes one token at a time
- Reuses cached Keys and Values
- Complexity: **O(T)** per step

Produces numerically equivalent outputs to full decoding.

---

## Logits Processing & Sampling

Logits are processed **before softmax** using a modular pipeline:

```
Logits
 → Repetition Penalty
 → Temperature Scaling
 → Top-k Filtering
 → Top-p Filtering
 → Softmax
 → Sampling
```

### Repetition Penalty

Previously generated tokens are penalized to prevent loops:

- Positive logits: divide by penalty
- Negative logits: multiply by penalty

---

## Correctness Validation

The implementation includes equivalence testing:

- Cached vs non-cached decoding
- Maximum difference ≈ **1e-7**

This confirms mathematical correctness of KV caching.

---

## Performance Benchmarking

Benchmarks compare:
- Full decoding
- Cached decoding

Results show increasing speedup with longer sequences, validating expected complexity reduction.

---

## Design Decisions & Trade-offs

- **Clarity over micro-optimizations**
- Explicit shape handling
- Modular decoding pipeline
- CPU-first benchmarking

The focus is correctness and understanding, not maximal throughput.

---

## Project Status & Extensions

### Completed
- Transformer inference engine
- KV cache per layer
- Incremental decoding
- Sampling & repetition penalty
- Correctness & performance tests

### Possible Extensions
- Batched decoding
- Streaming token generation
- KV cache memory profiling
- FlashAttention-style optimizations

---

## Conclusion

This project demonstrates how modern LLM inference works **under the hood**, with a clean, correct, and extensible implementation of KV-cached decoding.

It is intended as a learning resource and a systems-focused reference for Transformer inference engineering.

