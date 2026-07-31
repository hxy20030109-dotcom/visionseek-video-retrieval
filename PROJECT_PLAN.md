# VisionSeek Project Plan

## 1. Project Charter

### Objective

Build a reproducible bilingual text-to-video retrieval system and determine
whether a learned, budget-aware router can avoid unnecessary second-stage
reranking without giving up most of its accuracy improvement.

### Target roles

- Multimodal / video understanding algorithm intern
- Search, ranking, or recommendation algorithm intern
- Applied Machine Learning Engineer intern
- Machine Learning Engineer intern

The project is not presented as proof of low-level AI infrastructure expertise
or foundation-model research.

### Delivery window

- Start: 2026-07-31
- Feature complete: 2026-10-25
- Portfolio complete: 2026-11-08
- Interview and application buffer: 2026-11-09 to 2026-11-22

### Time budget

Target 18–22 focused hours per week:

- Four weekdays: 2 hours each
- One long weekday or weekend block: 4 hours
- One weekend block: 6–10 hours

## 2. Success Criteria

The project is complete only when all mandatory criteria are satisfied.

### Functional

- A text query retrieves ranked videos from a held-out corpus.
- English and Chinese queries use the same documented interface.
- The system supports no reranking, always reranking, rule-based routing, and
  learned routing.
- Every experiment is driven by a versioned configuration.

### Scientific

- At least three meaningful baselines are reproduced.
- At least four ablations isolate the contribution of routing and reranking.
- Routing thresholds and hyperparameters are selected without test-set leakage.
- Main training results are repeated with three seeds where computationally
  practical.

### Engineering

- Unit tests cover similarity, ranking metrics, and router features.
- One command runs a smoke test from clean embeddings to retrieval metrics.
- ONNX Runtime inference is benchmarked against eager PyTorch.
- The repository contains setup instructions, architecture documentation,
  experiment tables, limitations, and error analysis.

### Portfolio

- README contains only verified results.
- A two-minute demo video explains the problem, system, and measured trade-off.
- One architecture figure and one accuracy-latency Pareto plot are included.
- The full project can be explained without claiming ownership of third-party
  models or code.

## 3. Scope

### Mandatory

1. OpenCLIP frame-mean baseline on an MSR-VTT development subset.
2. Standard MSR-VTT English evaluation.
3. FAISS first-stage Top-K retrieval.
4. Lightweight temporal reranker with a frozen visual backbone.
5. Fixed-threshold router using score uncertainty.
6. Learned budget-aware router.
7. VATEX English/Chinese evaluation.
8. PyTorch versus ONNX Runtime latency benchmark.
9. Ablation study and qualitative error analysis.

### Stretch

- InternVideo2-S or another compact video encoder as a stronger backbone.
- Frame-narration branch using legally reusable/pre-generated descriptions.
- Hard-negative or hubness-aware loss inspired by recent retrieval work.
- FP16/INT8 TensorRT benchmark.
- Minimal local search demo.

### Explicitly out of scope

- Pretraining a video foundation model.
- Full fine-tuning of a billion-parameter model.
- A custom CUDA kernel before the core experiments are complete.
- A production-scale frontend or cloud platform.
- Claims of state-of-the-art performance.
- Combining every referenced repository into one environment.

## 4. Technical Strategy

### Stage 1: fast retrieval

- Sample a fixed number of frames per video.
- Encode frames with a frozen dual encoder.
- Mean-pool normalized frame embeddings for the first baseline.
- Store video embeddings in a FAISS inner-product index.
- Retrieve Top-K candidates for each text query.

### Stage 2: expensive reranking

- Preserve candidate frame tokens rather than only the pooled video vector.
- Use query-aware temporal attention to aggregate relevant frames.
- Optionally fuse frame-level narrations through a separate text branch.
- Train only the small reranking module first.

### Routing

Rule baseline:

- Route when the Top-1/Top-2 score margin is small.
- Compare margin, entropy, and score dispersion.

Learned router:

- Input: Top-K score statistics and query features.
- Label: whether reranking improves the query outcome on training/validation data.
- Objective: classification loss plus a tunable compute-cost penalty.
- Output: probability that reranking is worth its cost.

### Deployment

- Export only components supported reliably by ONNX.
- Benchmark fixed batch sizes first.
- Report preprocessing, encoder, retrieval, routing, and reranking latency
  separately.

## 5. Milestones and Gates

### Phase 0 — Foundation (2026-07-31 to 2026-08-02)

Deliverables:

- Independent repository and environment instructions
- Tested cosine-retrieval and routing-feature utilities
- Synthetic smoke test with deterministic output
- Project, architecture, experiment, and reference documentation

Gate G0:

- Tests and smoke test pass locally.
- GPU is detected by PyTorch.

### Phase 1 — Honest baseline (2026-08-03 to 2026-08-16)

Deliverables:

- MSR-VTT acquisition/preprocessing documentation
- Reproducible subset and full-split dataloaders
- OpenCLIP frame-mean baseline
- R@1/R@5/R@10, median rank, runtime, and memory report

Gate G1:

- One command reproduces the baseline from cached embeddings.
- First verified results can be placed in the README.
- Project may be listed on the résumé as “ongoing”.

### Phase 2 — Temporal reranker (2026-08-17 to 2026-08-30)

Deliverables:

- Candidate frame-token cache
- Small query-aware temporal module
- Training, validation, checkpoint, and evaluation pipeline
- Always-rerank comparison against Phase 1

Gate G2:

- Reranker beats the first-stage baseline on validation data.
- Improvement is verified on the held-out test split.

### Phase 3 — Budget-aware routing (2026-08-31 to 2026-09-13)

Deliverables:

- Margin/entropy threshold baselines
- Labels describing per-query reranking benefit
- Learned router and compute-cost objective
- Quality-versus-reranker-usage curve

Gate G3:

- Learned routing is compared fairly with always/never/rule routing.
- Failure cases are documented even if the hypothesis is rejected.

### Phase 4 — Bilingual evaluation (2026-09-14 to 2026-09-27)

Deliverables:

- VATEX English and Chinese query pipeline
- Per-language metrics and error categories
- Analysis of cross-language routing calibration

Gate G4:

- Dataset provenance and splits are documented.
- No machine-translated test labels are presented as official bilingual ground truth.

### Phase 5 — Stronger retrieval experiments (2026-09-28 to 2026-10-11)

Deliverables:

- One stronger encoder or one caption-aware extension
- Hard-negative/hubness experiment only if core work is stable
- Final ablation matrix

Gate G5:

- Every added component has a measurable contribution or is removed.

### Phase 6 — Inference optimization (2026-10-12 to 2026-10-25)

Deliverables:

- ONNX export and numerical parity checks
- FP32/FP16 benchmark
- p50/p95 latency, throughput, peak VRAM, and index-size report
- Accuracy-latency Pareto curve

Gate G6:

- Benchmarks include hardware, warmup, iterations, batch size, and precision.

### Phase 7 — Portfolio release (2026-10-26 to 2026-11-08)

Deliverables:

- Recruiter-readable README
- Architecture figure
- Demo script/video
- Model card, limitations, references, and reproducibility guide
- Tagged v1.0 release

Gate G7:

- A new user can run the smoke test from the README.
- Every résumé bullet is supported by a committed result.

### Phase 8 — Interview hardening (2026-11-09 to 2026-11-22)

Deliverables:

- Five-minute and fifteen-minute project explanations
- Answers to likely model, data, systems, and experiment questions
- Targeted résumé variants for multimodal, search/ranking, and ML engineering

## 6. Application Schedule

Do not wait for v1.0 before applying.

- 2026-08-17: add the verified baseline as an ongoing project.
- 2026-09-14: update with reranking and first routing results.
- 2026-10-26: replace with the full performance story.
- Apply continuously from mid-August; use each milestone as a reason to update
  recruiters and referrals.

## 7. Risk Register

| Risk | Early warning | Mitigation |
|---|---|---|
| Dependency conflicts | More than one day lost to legacy environments | Reimplement small ideas in one modern environment; isolate legacy scripts |
| GPU memory pressure | OOM with batch size one | Freeze backbone, cache features, reduce frames/resolution, use mixed precision |
| Dataset access delay | Raw videos unavailable by Day 5 | Begin with a documented subset and cached/open alternatives |
| Weak originality | Router is only a hand-tuned threshold | Add learned gain prediction and a compute-penalized objective |
| No measurable gain | Reranker does not improve validation recall | Treat as a valid result, debug by query category, simplify architecture |
| Test leakage | Test results influence thresholds | Freeze test evaluation; tune only on train/validation |
| Open-source license uncertainty | Referenced repository has no clear license | Use the paper/idea only; do not copy code or redistribute assets |
| Schedule slip | Two consecutive weekly gates missed | Drop stretch work, never drop evaluation or reproducibility |

## 8. Definition of Done

VisionSeek v1.0 is done when:

1. The mandatory scope is implemented.
2. All tests pass from a clean environment.
3. Dataset and third-party provenance are documented.
4. Baseline, always-rerank, rule-router, and learned-router results are reported.
5. English and Chinese evaluation is complete.
6. Accuracy and latency are measured under a reproducible protocol.
7. Limitations and negative results are stated honestly.
8. README, demo, release tag, and interview notes are ready.
