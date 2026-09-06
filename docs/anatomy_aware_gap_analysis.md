# Obsolete — safe to delete this file

This file was the pre-implementation audit of the old repository layout. Everything in it that
still matters — the unresolved public-encoder contracts, the missing manifests and label
columns, the EHR/PESI contract, the absent expert segmentation annotations — is recorded and
kept current in [BLOCKERS.md](BLOCKERS.md).

Current documentation:

- [PIPELINE.md](PIPELINE.md) — the active pipeline, stage by stage
- [EXPERIMENT_MAP.md](EXPERIMENT_MAP.md) — every experiment, plus the old-id → new-id mapping
- [BLOCKERS.md](BLOCKERS.md) — every unresolved data/checkpoint contract

It could not be deleted automatically because the file was locked by another process. Delete
it manually:

```powershell
Remove-Item -Force E:\PE_NU\source\pe-project\docs\anatomy_aware_gap_analysis.md
```
