# 2107004 result consolidation

The original result batch remains `ep001` through `ep006`. The imported batch was renumbered
from its original `ep001` through `ep005` to `ep007` through `ep011`:

| Source batch ID | Canonical ID | Available stages |
| --- | --- | --- |
| `ep001` | `ep007` | diarization, ASR, fused output, annotation draft |
| `ep002` | `ep008` | diarization, ASR, fused output, annotation draft |
| `ep003` | `ep009` | diarization, ASR, fused output, annotation draft |
| `ep004` | `ep010` | diarization, ASR, fused output, annotation draft |
| `ep005` | `ep011` | diarization only |

`raw/registry.csv`, `raw/episode_metadata.csv`, and `raw/episode_ids.json` now describe all
eleven recordings. `episode_id_map.csv` is the machine-readable mapping and stage-coverage list.

No files were discarded. The unmodified imported tree is retained in
`_source_runs/new_results_original/`; the prior raw metadata and aggregate outputs are retained
under `_source_runs/`.

The five new source audio files and processed WAV files were not present in the supplied
`new_results` tree, so they were not fabricated or substituted during consolidation.
