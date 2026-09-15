# 2107015 result consolidation

The original result batch remains `ep001` through `ep005`. The imported batch was renumbered
from its original `ep001` through `ep005` to `ep006` through `ep010`:

| Source batch ID | Canonical ID | Available stages |
| --- | --- | --- |
| `ep001` | `ep006` | diarization, ASR, fused output, annotation draft |
| `ep002` | `ep007` | diarization, ASR, fused output, annotation draft |
| `ep003` | `ep008` | diarization, ASR, fused output, annotation draft |
| `ep004` | `ep009` | diarization only |
| `ep005` | `ep010` | diarization only |

`data/raw/registry.csv`, `data/raw/episode_metadata.csv`, and `data/raw/episode_ids.json` now
describe all ten recordings. `data/episode_id_map.csv` is the machine-readable mapping and
stage-coverage list.

No files were discarded. The unmodified imported tree is retained in
`_source_runs/new_results_original/`; the prior raw metadata and aggregate outputs are retained
under `_source_runs/`.

The supplied `new_results` tree contained no source audio or processed WAV files, so none were
fabricated or substituted during consolidation.
