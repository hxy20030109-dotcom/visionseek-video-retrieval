# Data

Datasets are not committed to Git.

Planned datasets:

1. **MSR-VTT** — English development and standard text-to-video retrieval.
2. **VATEX** — final English/Chinese evaluation using official captions and splits.

Expected local layout:

```text
data/
├── raw/
│   ├── msrvtt/
│   └── vatex/
├── interim/
│   ├── frames/
│   └── metadata/
└── processed/
    ├── embeddings/
    └── indexes/
```

Before downloading or redistributing anything, record:

- official source URL;
- dataset license/terms;
- checksum or release version;
- exact train/validation/test split;
- any missing or corrupt videos;
- preprocessing command and configuration.

The test split must never be used to choose thresholds or checkpoints.
