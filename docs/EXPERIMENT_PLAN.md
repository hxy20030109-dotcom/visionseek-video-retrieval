# Experiment Plan

## Primary hypothesis

A learned router can reduce the proportion of queries sent to the temporal
reranker while preserving most of the retrieval improvement obtained by always
reranking.

## Baselines

| ID | System |
|---|---|
| B0 | Frozen dual encoder + mean-pooled frames + exact cosine retrieval |
| B1 | B0 + FAISS first-stage retrieval |
| B2 | B1 + always-on temporal reranker |
| B3 | B2 + fixed margin/entropy router |
| B4 | B2 + learned budget-aware router |
| B5 | B4 + optional frame-narration branch |

## Main metrics

Quality:

- Recall@1, Recall@5, Recall@10
- median rank and mean rank

Cost:

- reranker invocation rate
- mean, p50, and p95 end-to-end latency
- queries per second
- peak allocated GPU memory
- offline feature-extraction time
- index size

## Router labels

For a training or validation query, define reranking benefit from the change in
rank of its relevant video. Candidate labels to compare:

1. Binary: reranking moves the relevant video into Top-K.
2. Binary: reranking improves reciprocal rank by at least a fixed amount.
3. Regression: exact change in reciprocal rank.

The test split is never used to create labels, select a label definition, or
calibrate a threshold.

## Mandatory ablations

1. Mean pooling versus query-aware temporal aggregation.
2. Margin only versus all router features.
3. Rule-based versus learned routing at matched invocation rates.
4. Different reranking budgets: 10%, 25%, 50%, 75%, and 100%.
5. English versus Chinese performance and calibration.
6. PyTorch FP32 versus supported optimized inference modes.

## Statistical protocol

- Fix and log all random seeds.
- Use three seeds for learned lightweight modules when practical.
- Report mean and standard deviation.
- Keep preprocessing, corpus, and candidate count fixed in comparisons.
- Log failed runs and negative findings instead of deleting them.

## Performance protocol

- Record GPU, driver, PyTorch, CUDA runtime, and operating system.
- Include warmup before timing.
- Synchronize CUDA around measured regions.
- Use at least 100 timed online queries after warmup.
- State batch size, Top-K, number of frames, precision, and index type.
- Verify numerical/ranking parity after export.

## Result-table template

| System | R@1 | R@5 | R@10 | p50 ms | p95 ms | QPS | Rerank % | VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| B2 | TBD | TBD | TBD | TBD | TBD | TBD | 100 | TBD |
| B3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

No value replaces `TBD` until a committed script and configuration reproduce it.
