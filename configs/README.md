# `configs/` — configuration layout

```text
configs/
├── experiments.yaml    the registry: semantic name -> config, question, requirements, status
├── components/         reusable fragments; never run directly
│   ├── backbone/         registry.yaml holds all three contracts; ct_fm/ct_clip/totalfm select one
│   ├── encoder/          which checkpoint the encoder starts from (pretrained/dapt/c0/silver)
│   ├── dapt/             self-supervised objectives (none, MAE, DINO, SimCLR, anatomy)
│   ├── alignment/        image-report objective
│   ├── task/             task and cohort contracts (diagnosis, prognosis, organ silver
│   │                     supervision, anatomy reference architecture, ROI students, ...)
│   ├── fusion/           concat_mlp / late_logit / soft_moe
│   ├── adapter/          per-branch organ adapters
│   ├── silver/           SL00 / SL01 / SL02 provider settings
│   └── training/         optimization + evaluation contracts per family, incl. the probe
├── runs/               one file per runnable experiment
│   ├── 00_data/           dataset build, segmentation, ROI, silver generation
│   ├── 01_foundation/     frozen public-encoder probes + the zero-shot contract
│   ├── 02_representation/ DAPT, alignment, probes, RSPECT transfer, silver adaptation
│   ├── 03_diagnosis/      baseline/ and anatomy/
│   ├── 04_prognosis/      modality/, global/, anatomy/
│   ├── 05_anatomy_analysis/ counterfactual/ and students/{gt,kd}/
│   └── 90_deferred/       contour, concept bottleneck
├── compute/            GPU default or CPU
├── paths.yaml          data/output roots
```

## The two rules

1. **Only files under `runs/` are runnable.** They carry `experiment.id`, `experiment.stage`
   and `experiment.name`. Fragments under `components/` are inherited, never launched.
2. **A run config inherits only from `components/` — never from another run config.** Two
   levels, always. The old deep chains (`DX17` → `DX16` → `DX01` → four fragments) are what
   made this repository hard to read.
3. **Ids describe their output.** `experiment.id` names the run directory, so it must say what
   is in it: `SEG_pseudo_anatomy`, not `SEG01`.

So a run config shows exactly the axis it changes:

```yaml
_base_:
  - ../../../paths.yaml
  - ../../../compute/default.yaml
  - ../../../components/backbone/ct_fm.yaml
  - ../../../components/task/diagnosis.yaml
  - ../../../components/task/diagnosis_organ_silver.yaml
  - ../../../components/training/diagnosis.yaml
  - ../../../components/adapter/organ.yaml
  - ../../../components/fusion/soft_moe.yaml
experiment:
  id: DX_anatomy_silver_soft_moe
  name: diag.anatomy.silver.moe
  stage: diagnosis
  family: diagnosis
# ... only what this arm changes
```

Bases merge in order, later mappings win, lists and scalars are replaced whole (never merged
implicitly). `_replace_: true` inside a mapping discards the inherited mapping; `_delete_`
removes keys. Base paths resolve relative to the file that lists them.

## Names and ids

The semantic `experiment.name` (`diag.anatomy.silver.moe`) is the human interface and the
registry key. `experiment.id` (`DX_anatomy_silver_soft_moe`) is the output key: it names the
run directory under the output root and is recorded in every checkpoint's lineage, so it must
be self-describing — reading `outputs/diagnosis/DX_anatomy_silver_soft_moe/` should tell you
what produced it without opening a config.

Ids must start with the prefix their stage requires (`DX` diagnosis, `PR` prognosis, `D` dapt,
`AL` alignment, `SE` silver adaptation, `SL` silver, `SEG` segmentation, `ROI` roi,
`CF` counterfactual, `RS`/`KD` roi_student, `F` foundation, `CT` contour) —
`source/utils/config.py` enforces this. After the prefix, use lowercase words separated by
underscores that name the configuration, not a serial number.

The pre-refactor ids (`SEG01`, `DX18_ANATOMY_FULL`, `CF02_REMOVE_PA`, …) live on as
`legacy_id` in `configs/experiments.yaml` and in
[../docs/EXPERIMENT_MAP.md](../docs/EXPERIMENT_MAP.md), so old notes and result folders can
still be traced.

Never reuse one id for two scientific configurations.

## Adding an experiment

1. Copy the closest existing run config, change only the axis you are testing.
2. Give it a unique `experiment.id` (correct stage prefix) and a semantic `experiment.name`.
3. Add a registry entry in `configs/experiments.yaml`: config path, group, description,
   question, `requires`, `produces`, `status`, and `blockers` if a contract is unresolved.
4. Add its row to `docs/EXPERIMENT_MAP.md`.
5. `python run.py show <name>` then `python run.py preflight <name> --gpus 0`.

If the change is large enough to be a different scientific experiment, write a new run config
rather than a long chain of `--set` overrides that nobody can trace.

## The three config variables that make experiments comparable

### `data.profile` — which cohort

`test_500_sample` or `full_inspect`. Both are built by the same code from the same rules
(`source/dataset/profiles/`); only sampling differs. Defaults to `full_inspect` when a config
does not say. Manifest paths resolve under
`${PE_CLOUD_ROOT}/data/derived/datasets/<profile>/`, so the two cohorts never mix.

### `model.backbone` — which encoder architecture

`configs/components/backbone/registry.yaml` holds one contract per backbone (repo, checkpoint,
factory, output adapter, feature dim). The per-backbone files only *select* one, and carry the
registry with them, so `--set model.backbone=ct_clip` works from any run config. The registry
is dropped from the resolved config, so editing one backbone's contract does not change every
other experiment's config hash.

### `encoder.init_source` — which weights the encoder starts from

`pretrained` | `dapt` | `c0` | `silver`. The resolver turns that into
`lineage.source_checkpoint` (or `init.checkpoint` for the silver-adaptation stage),
`lineage.initialization`, `lineage.dapt`, `lineage.alignment` and
`lineage.encoder_init_source`. The task model itself never changes — that is the point.

```bash
python run.py run diag.anatomy.concat --gpus 0 --set encoder.init_source=dapt
```

Defaults come from `components/encoder/sources.yaml`, which points at the canonical run of
each stage. Any other checkpoint is named explicitly with `encoder.checkpoint` +
`encoder.source_experiment`; switching `model.backbone` without doing so is refused, because
the default DAPT/C0/C_silver checkpoints belong to one backbone.

### Run ids are stamped, so nothing collides

Each of the three variables records its baseline in the component that sets it
(`experiment.baseline_dataset` / `baseline_backbone` / `baseline_init`). Only a deviation is
appended to `experiment.id`:

```text
DX_anatomy_concat                                              baseline
DX_anatomy_concat__enc_dapt                                    DAPT initialization
DX_anatomy_concat__ds_test_500_sample__bb_ct_clip__enc_silver  all three changed
```

A config left at its defaults keeps exactly the id it had before. Set
`experiment.variant_stamp: false` to opt out, and then own the collision yourself.

## Small overrides

```bash
python run.py run probe.diag --gpus 0 --set encoder.init_source=dapt
```

Overrides are applied before environment expansion, so `${PE_CLOUD_ROOT}` works inside them.
Overriding `data.profile`, `model.backbone` or `encoder.init_source` stamps the run id for
you. Overriding `lineage.source_checkpoint` directly does not — if you go around the encoder
resolver, override `experiment.id` yourself or two runs will write to the same directory.

## Compute

`compute/default.yaml` is one GPU (`strategy: auto`, `devices: [0]`); `--gpus` overrides
devices/strategy/accelerator. `compute/cpu.yaml` is for the dataset build and for ROI
construction, which parallelize over CPU workers rather than DDP.

A batch sweep across GPU groups is not a config in this tree: write a `parallel:` file
wherever you like and pass it to `tools/launch_parallel.py` (schema in the root README).

## Unresolved contracts

Several fields are intentionally empty — backbone factory/output_adapter/feature_dim, report
embedding columns, the EHR column lists, the external cohort manifest. Preflight fails until
they are filled from the real artifacts. Do not guess them; see
[../docs/BLOCKERS.md](../docs/BLOCKERS.md).

## No second copy of the pipeline

`runs/` is the only place a runnable experiment lives. Every arm that is scientifically
relevant was carried into it, so there is no parallel set of older configs to keep in sync —
the pre-refactor ids survive only as `legacy_id` documentation.
