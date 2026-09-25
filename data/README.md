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

## Small local-video experiment

Place 5-10 short, self-owned videos in:

```text
data/raw/local_videos/
```

Then create the local metadata and annotation manifest:

```powershell
.\.venv\Scripts\python.exe scripts\import_local_videos.py
```

The generated `data/interim/local_videos_manifest.csv` contains video metadata
plus blank `query_en` and `query_zh` columns. Add one accurate English and one
accurate Chinese description for every video. Raw videos and generated
manifests remain local and are not committed to Git.
