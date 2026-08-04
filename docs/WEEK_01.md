# Week 1 Execution Plan

Dates: 2026-07-31 to 2026-08-07

## Weekly outcome

Finish the engineering foundation and understand the complete tensor flow of a
small retrieval system before touching a large video model.

## Day 0 — 2026-07-31

- [x] Freeze the project question and scope.
- [x] Create repository structure and reproducibility rules.
- [x] Implement similarity, retrieval metrics, and router features.
- [x] Create a clean local Python environment.
- [x] Install CUDA-enabled PyTorch.
- [x] Run environment check, unit tests, lint, and synthetic smoke test.
- [ ] Explain every tensor shape in the smoke test without reading the answer.

Acceptance:

- `pytest` passes.
- `python scripts/smoke_test.py` reports high synthetic Recall@1.
- CUDA identifies the RTX 5070 Ti.

Verified on 2026-07-31:

- PyTorch 2.12.1+cu130
- CUDA available
- NVIDIA GeForce RTX 5070 Ti, 15.9 GiB visible to PyTorch
- 8 unit tests passed
- Ruff passed
- synthetic Recall@1 = 100% (sanity check only, not a portfolio result)

## Day 1

- Review `torch.Tensor`, shapes, broadcasting, matrix multiplication, softmax,
  sorting, `topk`, and normalization.
- Rewrite the similarity matrix once without looking at the project code.
- Add one test for a deliberately incorrect embedding shape.

## Day 2

- Implement a tiny `Dataset` and `DataLoader` with synthetic query/video pairs.
- Batch queries and verify that metric results match the unbatched version.
- Write a short note explaining why cosine similarity uses normalized vectors.

## Day 3

- [x] Read the OpenCLIP model-loading and preprocessing interface.
- [x] Run one image-text example on the GPU.
- [x] Verify image and text embeddings have shape `[batch, 512]`.

## Day 4

- [x] Generate two synthetic MP4 files for a controlled smoke test.
- [x] Sample four frames from each video with a reusable project module.
- [x] Encode frames with OpenCLIP and mean-pool them into video embeddings.
- [ ] Repeat the pipeline with two short, self-owned videos.
- [ ] Compare mean pooling with single-frame retrieval.
- [x] Keep all sample media outside Git unless redistribution is permitted.

## Day 5+

- Prepare the MSR-VTT access checklist and metadata schema.
- Implement dataset validation before downloading the full corpus.
- Run the baseline on the smallest legal subset available.
- Write the first weekly report: what worked, what failed, and next actions.

## Concepts that must be explainable by the end of the week

- Why matrix multiplication gives all query-video similarities.
- Why embeddings are normalized.
- Difference between retrieval Top-K and classification Top-K.
- What Recall@K measures.
- Why test data cannot select a router threshold.
- What freezing a backbone changes during backpropagation.
