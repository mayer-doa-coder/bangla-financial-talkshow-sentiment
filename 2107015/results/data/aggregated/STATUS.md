# Aggregate outputs need regeneration

The old and new batches each contained aggregate files calculated for their own source run.
They cannot represent the combined ten-episode dataset, so both unmodified versions are kept in
`../../_source_runs/previous_aggregated/` and
`../../_source_runs/new_results_original/aggregated/`.

Regenerate aggregate metrics and submission gates after the combined episode set is complete.
