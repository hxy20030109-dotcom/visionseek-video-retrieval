# Architecture

## Design principle

The system separates fast corpus-scale retrieval from expensive candidate-level
reasoning. The router is useful only if it produces a measurable quality-cost
trade-off.

## Data flow

```text
Video preprocessing
raw video -> sampled frames -> frozen frame/video encoder
          -> pooled embedding + optional frame-token cache

Indexing
pooled video embeddings -> L2 normalization -> FAISS inner-product index

Query
text -> text encoder -> normalized query embedding
     -> FAISS Top-K candidates
     -> confidence feature extractor
     -> router
         -> confident: first-stage ranking
         -> uncertain: query-aware temporal reranker
     -> final ranking
```

## Modules

### Encoder

Phase 1 uses a frozen OpenCLIP model because it is easy to debug and establishes
an honest baseline. A compact InternVideo model is evaluated later only if it
fits the compute budget.

### First-stage representation

Each video is represented by the normalized mean of its normalized frame
embeddings. This deliberately simple baseline makes later improvements
interpretable.

### Temporal reranker

For each Top-K candidate, the reranker receives:

- query embedding;
- ordered frame embeddings;
- frame mask;
- optional frame-narration embeddings.

The first implementation uses query-conditioned attention over frames followed
by a small scoring head. The foundation encoder remains frozen.

### Router

The router receives statistics from the first-stage Top-K scores:

- Top-1 score;
- Top-1/Top-2 margin;
- entropy of the Top-K softmax distribution;
- score standard deviation;
- optional query-length/language features.

Two policies are compared:

1. A transparent fixed-threshold policy.
2. A learned MLP predicting whether reranking improves the query.

The learned policy is calibrated on validation data for different compute
budgets.

## Interfaces

Initial tensor contracts:

- query embeddings: `[batch, embedding_dim]`
- video embeddings: `[num_videos, embedding_dim]`
- frame embeddings: `[batch, candidates, frames, embedding_dim]`
- first-stage scores: `[batch, num_videos]`
- candidate scores: `[batch, candidates]`
- router features: `[batch, num_features]`

These contracts must be tested before model complexity is added.

## Deployment boundary

Feature extraction and online querying are benchmarked separately. The final
latency report distinguishes:

1. text preprocessing and encoding;
2. FAISS retrieval;
3. router inference;
4. optional reranking;
5. result serialization.
