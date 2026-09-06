# Obsolete — safe to delete this file

This file described the pre-refactor repository: opaque experiment ids (`DX18_ANATOMY_FULL`,
`CF02_REMOVE_PA`, …), a flat `configs/experiment/` directory, and commands pointing at configs
that no longer exist. Keeping it would only mislead.

Current documentation:

- [PIPELINE.md](PIPELINE.md) — the active pipeline, stage by stage
- [EXPERIMENT_MAP.md](EXPERIMENT_MAP.md) — every experiment, plus the old-id → new-id mapping
- [BLOCKERS.md](BLOCKERS.md) — every unresolved data/checkpoint contract
- [../README.md](../README.md) — install, environment, and the `run.py` workflow

It could not be deleted automatically because the file was locked by another process. Delete
it manually:

```powershell
Remove-Item -Force E:\PE_NU\source\pe-project\docs\anatomy_aware_implementation.md
```
