# References and Reuse Policy

This file records external work before any implementation is adapted.

| Resource | Intended use | Reuse rule |
|---|---|---|
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Initial dual-encoder baseline | Use as a dependency; preserve its license and attribution |
| [InternVideo](https://github.com/OpenGVLab/InternVideo) | Strong video-encoder benchmark | Use released APIs/checkpoints under their stated terms; do not vendor the repository |
| [FAISS](https://github.com/facebookresearch/faiss) | First-stage vector index | Use as a dependency and cite the project |
| [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) | Historical retrieval protocol reference | Study metrics/splits; independently implement the current pipeline |
| [NarVid](https://github.com/invhun/NarVid) | Frame-narration and reranking research reference | Do not copy code unless a compatible license is confirmed; verify data terms separately |
| [NeighborRetr](https://github.com/zzezze/NeighborRetr) | Optional hubness/hard-negative research reference | Use only after the core router is complete; attribute adapted ideas |
| [ONNX Runtime examples](https://github.com/microsoft/onnxruntime-inference-examples) | Export and benchmark patterns | Adapt small examples with attribution where required |
| [Torch-TensorRT](https://github.com/pytorch/TensorRT) | Optional deployment benchmark | Use as a dependency; preserve license notices |

## Attribution procedure

Before adapting a source file:

1. Check its repository license.
2. Record the exact source URL and commit.
3. Keep the original copyright notice when required.
4. Mark modifications in the file header.
5. Add the source to the final report.

Research ideas may be reimplemented from papers, but they must still be cited.
Released captions, datasets, and checkpoints have separate terms that must be
checked before redistribution.
