"""RADAR side of tools/tasks/zeroshot_radar.py. Runs inside the RADAR conda env, not the project env.

The RADAR repository pins transformers==4.25 and timm==0.4.12, which cannot share an
environment with the project, so zeroshot_radar.py launches this file with the RADAR env's
Python and the two exchange plain JSON: a job file in, one JSON line per study out. Nothing
here imports the project.

    <radar-python> tools/tasks/zeroshot_radar_worker.py --job <run>/.radar_job.json

Per study it reproduces ``RADAR_inference/inference_demo.py`` (the released inference path)
for a single organ-level finding: resample to 1 x 1 x 5 mm, clip [-300, 400] HU, min-max
scale, crop the non-zero box, pad to 96 x 256 x 384, slide 96 x 256 x 384 windows with 25%
overlap, let RADAR's own segmentation head find the organ, and score the organ token against
a [negative, positive] text pair; when the organ touches every window border the demo's
centre-crop fallback is used. The demo reads files as stored; INSPECT and the demo case are
both LAS, and anything else is reoriented to LAS first.

The text pair follows the shipped ``infer_text_embedding_radar.pt`` convention, checked by
re-encoding: the negative is the training default caption "normal." (cosine 0.98 to every
shipped negative) and the positive is the mean of several unit-normalised report-style
sentences, left unnormalised (shipped positives have norm ~0.77).

Job modes: ``score`` appends {"study_id", "y_prob", ...} or {"study_id", "error"} lines to
``scores_path`` and flushes each one, so an interrupted run resumes; ``preview`` re-runs the
listed studies and writes one PNG each (preprocessed input + RADAR's organ mask).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROI_SIZE = (96, 256, 384)
OVERLAP = 0.25
REFERENCE_SPACING_MM = (1.0, 1.0, 5.0)     # (x, y, z) of the stored array
CLIP_HU = (-300.0, 400.0)
CROP_MARGIN = (5, 20, 20)                  # (d, h, w) voxels around the non-zero box
MAX_SIDE = 1000                            # the demo skips larger resampled volumes
SEG_CLASSES = 37                           # background + 36 organs


def _log(message: str) -> None:
    print(message, flush=True)


def load_radar(repo: Path, checkpoint: Path, device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    """Build the demo's RADAR model and load a released checkpoint."""
    ckpt_root = repo / "ckpt"
    # inference_demo reads these at import time for the tokenizer and BERT config.
    os.environ["MODEL_ROOT"] = str(ckpt_root)
    os.environ["CONFIGS_ROOT"] = str(ckpt_root)
    sys.path.insert(0, str(repo / "RADAR_inference"))
    import inference_demo as radar
    from dynamic_network_architectures.med import XBertEncoder
    from dynamic_network_architectures.vision_branch import VisionBranch

    model = radar.RADAR(image_encoder=VisionBranch(), text_encoder=XBertEncoder.from_config({}, from_pretrained=True))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["model"]
    message = model.load_state_dict(state, strict=False)
    # cls.predictions is BERT's masked-LM head: never used to embed text.
    missing = [key for key in message.missing_keys if ".cls.predictions." not in key]
    if missing:
        raise RuntimeError(f"RADAR checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")
    model.eval().to(device)
    return radar, model, {"unexpected_keys": len(message.unexpected_keys), "temperature": float(model.temp)}


@torch.no_grad()
def text_pair(model: torch.nn.Module, negatives: list[str], positives: list[str], device: torch.device) -> torch.Tensor:
    """[negative, positive] rows in the space of the shipped infer_text_embedding_radar.pt."""
    tokens = model.tokenizer(negatives + positives, padding="max_length", truncation=True,
                             max_length=100, return_tensors="pt").to(device)
    hidden = model.text_encoder.forward_text(tokens).last_hidden_state
    rows = F.normalize(model.text_proj(hidden[:, 0, :]), dim=-1)
    return torch.stack([rows[: len(negatives)].mean(0), rows[len(negatives):].mean(0)])


def preprocess(path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """RADAR demo input for one NIfTI: [1, D, W, H] in [0, 1]."""
    import nibabel as nib
    from monai import transforms

    image = nib.load(path)
    axcodes = nib.aff2axcodes(image.affine)
    if axcodes != ("L", "A", "S"):
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(("L", "A", "S"))
        )
        image = image.as_reoriented(transform)
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    volume = torch.from_numpy(np.asanyarray(image.dataobj, dtype=np.float32))[None]   # [1, H, W, D]
    _, h, w, d = volume.shape
    scale = [spacing[i] / REFERENCE_SPACING_MM[i] for i in range(3)]
    # H, W, D are the stored (x, y, z) = (L, A, S) axes, so each takes its own spacing. The
    # demo pairs H with the y spacing and W with x; that is identical for the square in-plane
    # pixels of axial CT and only differs (wrongly) for anisotropic in-plane spacing.
    target = [int(h * scale[0]), int(w * scale[1]), int(d * scale[2])]
    volume = transforms.Resize(spatial_size=target, mode="trilinear")(volume)
    if hasattr(volume, "as_tensor"):                                             # MONAI MetaTensor
        volume = volume.as_tensor()
    volume = volume.permute(0, 3, 2, 1).contiguous()                             # [1, D, W, H]
    volume = volume.clamp(*CLIP_HU)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    nonzero = torch.nonzero(volume[0])
    low = torch.clamp(nonzero.min(0).values - torch.tensor(CROP_MARGIN), min=0)
    high = torch.minimum(nonzero.max(0).values + torch.tensor(CROP_MARGIN), torch.tensor(volume.shape[1:]))
    volume = volume[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    pad = [max(0, size - current) for size, current in zip(ROI_SIZE, volume.shape[1:])]
    volume = F.pad(volume, (0, pad[2], 0, pad[1], 0, pad[0]))                    # method="end", value 0
    return volume, {"orientation": "".join(axcodes), "spacing_mm": [round(v, 4) for v in spacing],
                    "input_shape": list(volume.shape[1:])}


@torch.no_grad()
def score_volume(radar: Any, model: torch.nn.Module, volume: torch.Tensor, item: str, organ: str,
                 text_feat: torch.Tensor, device: torch.device, keep_mask: bool = False) -> dict[str, Any]:
    """The demo's evaluate() loop for one finding on one volume."""
    from monai import transforms
    from monai.data.utils import dense_patch_slices

    if max(volume.shape[1:]) > MAX_SIDE:
        raise ValueError(f"resampled volume {list(volume.shape[1:])} exceeds {MAX_SIDE} voxels")
    organ_id = model.organs.index(organ)
    text_feat_dict = {item: text_feat}
    image = volume[None].to(device)                                              # [1, 1, D, W, H]
    size = list(image.shape[2:])
    slices = dense_patch_slices(size, ROI_SIZE, radar._get_scan_interval(size, ROI_SIZE, 3, OVERLAP))
    logits: dict[str, list] = {item: []}
    organ_feats: dict[str, Any] = {}
    stitched = torch.zeros((1, SEG_CLASSES, *size), device=device)
    counts = torch.zeros((1, 1, *size), device=device)
    for window in slices:
        index = (slice(0, 1), slice(None), *window)
        patch = image[index]
        logits, seg_prob = model.forward_test_win(patch, None, logits, [organ], text_feat_dict, organ_feats, None)
        stitched[index] += F.interpolate(seg_prob, size=patch.shape[2:], mode="trilinear")
        counts[index] += 1
    labels = (stitched / counts.clamp(min=1)).argmax(1).unsqueeze(0)              # [1, 1, D, W, H]
    del stitched
    organ_mask = labels == organ_id + 1
    organ_voxels = int(organ_mask.sum())
    fallback = False
    if not logits[item] and organ_voxels:
        # The organ touched a border in every window: rescore on a crop centred on it.
        fallback = True
        patch, patch_mask = radar.center_crop(image, organ_mask, crop_size=ROI_SIZE)
        patch_mask = patch_mask.float()
        patch_mask[patch_mask == 1] = organ_id + 1
        padded = transforms.DivisiblePadd(keys=["image", "label"], k=32, mode="constant", method="end")(
            {"image": patch[0], "label": patch_mask[0]}
        )
        logits, _ = model.forward_test_win(padded["image"][None], None, logits, [organ], text_feat_dict,
                                           organ_feats, None, skip_organ=organ_id)
    result: dict[str, Any] = {"windows": len(slices), "organ_voxels": organ_voxels, "fallback_crop": fallback}
    if not logits[item] and not organ_voxels:
        result["error"] = f"RADAR's segmentation found no {organ} voxels, so there is no organ token to score"
    elif not logits[item]:
        result["error"] = f"{organ} found ({organ_voxels} voxels) but never intact, even in the centre crop"
    else:
        result["y_prob"] = float(np.concatenate(logits[item]).mean(0)[1])
    if keep_mask:
        result["mask"] = organ_mask[0, 0].cpu().numpy()
    return result


def render_preview(path: Path, volume: torch.Tensor, mask: np.ndarray, study: dict[str, Any],
                   probability: float | None, threshold: float, threshold_rule: str, organ_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = volume[0].numpy()                                                    # [D, W(y), H(x)], LAS
    area_axial = mask.sum(axis=(1, 2))
    area_coronal = mask.sum(axis=(0, 2))
    z = int(area_axial.argmax()) if area_axial.any() else image.shape[0] // 2
    y = int(area_coronal.argmax()) if area_coronal.any() else image.shape[1] // 2
    # Radiological display: anterior up and patient right on the left (x grows to the left).
    axial, axial_mask = image[z][::-1], mask[z][::-1]
    coronal, coronal_mask = image[:, y][::-1], mask[:, y][::-1]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for axis, picture, overlay, aspect, name in (
        (axes[0], axial, axial_mask, 1.0, f"axial slice {z + 1}/{image.shape[0]}"),
        (axes[1], coronal, coronal_mask, REFERENCE_SPACING_MM[2], f"coronal row {y + 1}/{image.shape[1]}"),
    ):
        axis.imshow(picture, cmap="gray", vmin=0, vmax=1, aspect=aspect)
        axis.imshow(np.ma.masked_where(~overlay, overlay), cmap="autumn", alpha=0.45, aspect=aspect)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    p_text = "n/a" if probability is None else f"{probability:.4f}"
    figure.suptitle(
        f"RADAR zero-shot | {study['patient_id']} / {study['study_id']} | {study['outcome']}\n"
        f"GT={study['y_true']}  pred={study['y_pred']}  p={p_text}  threshold={threshold:.4f} ({threshold_rule})\n"
        f"overlay = RADAR's own {organ_label} segmentation ({int(mask.sum())} voxels at 1x1x5 mm); "
        f"input = model tensor after [-300, 400] HU clip and min-max scaling",
        fontsize=10,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    device = torch.device(job["device"])
    started = time.perf_counter()
    radar, model, info = load_radar(Path(job["repo"]), Path(job["checkpoint"]), device)
    text_feat = text_pair(model, job["prompts"]["negative"], job["prompts"]["positive"], device)
    _log(f"model ready device={device} temperature={info['temperature']:.4f} "
         f"text_pair_norms={[round(float(v), 3) for v in text_feat.norm(dim=-1)]} "
         f"load_sec={time.perf_counter() - started:.1f}")
    item, organ, studies = job["item"], job["organ"], job["studies"]

    with ThreadPoolExecutor(max_workers=1) as loader:                              # overlap I/O and GPU
        pending = loader.submit(preprocess, studies[0]["path"]) if studies else None
        began = time.perf_counter()
        for position, study in enumerate(studies, start=1):
            future = pending
            pending = loader.submit(preprocess, studies[position]["path"]) if position < len(studies) else None
            record: dict[str, Any] = {"study_id": study["study_id"]}
            try:
                volume, meta = future.result()
                outcome = score_volume(radar, model, volume, item, organ, text_feat, device,
                                       keep_mask=job["mode"] == "preview")
                record.update({key: value for key, value in outcome.items() if key != "mask"}, **meta)
                if job["mode"] == "preview":
                    render_preview(Path(study["png"]), volume, outcome["mask"], study, outcome.get("y_prob"),
                                   float(job["threshold"]), str(job["threshold_rule"]), job["organ_label"])
                    _log(f"preview {position}/{len(studies)} {study['study_id']} -> {study['png']}")
            except Exception as exc:  # noqa: BLE001 - one bad series must not stop the cohort
                record["error"] = f"{type(exc).__name__}: {exc}"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if job["mode"] == "score":
                with open(job["scores_path"], "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if "error" in record:
                    _log(f"study {study['study_id']} skipped: {record['error']}")
                if position % 25 == 0 or position == len(studies):
                    rate = (time.perf_counter() - began) / position
                    _log(f"scored {position}/{len(studies)} ({rate:.1f} s/study, "
                         f"~{rate * (len(studies) - position) / 3600:.1f} h left)")
            elif "error" in record:
                _log(f"preview {study['study_id']} failed: {record['error']}")
    _log(f"worker finished mode={job['mode']} studies={len(studies)} elapsed_sec={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
