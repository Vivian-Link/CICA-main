# Result records

`reported_metrics.csv` and `reported_table_rows.csv` contain the existing manuscript values, labelled `MANUSCRIPT_REPORTED_REFERENCE`. They are reference records, not outputs from a new training run. Empty cells denote unreported quantities. No per-seed predictions, checkpoints, missing standard deviations or new benchmark scores are inferred from aggregate values.

Actual training outputs are written to the selected run directory with dataset hashes, resolved settings, checkpoint selection, per-seed metrics and measured sample standard deviations. `SMOKE_TEST` outputs are software checks on generated fixtures and must be kept separate from MOSI/MOSEI benchmark records.
