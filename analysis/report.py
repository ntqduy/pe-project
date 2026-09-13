"""Render the EDA summary as a readable Markdown report."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _inventory(summary: Mapping[str, Any]) -> list[str]:
    rows = summary.get("inventory") or []
    if not rows:
        return []
    lines = ["## Artifact có sẵn", "", "| Bảng | Trạng thái | Dòng | Cột | Đường dẫn |", "|---|---|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| `{row['table']}` | {row['status']} | {row['rows']} | {row['columns']} | `{row['path']}` |"
        )
    return [*lines, ""]


def _cohort(summary: Mapping[str, Any]) -> list[str]:
    cohort = summary.get("cohort") or {}
    manifests = cohort.get("per_manifest") or {}
    if not manifests:
        return []
    lines = ["## Cohort", "", "| Manifest | Patient | Study | Study/patient (min–median–max) | Patient nhiều study |", "|---|---:|---:|---|---:|"]
    for name, values in manifests.items():
        per = values["studies_per_patient"]
        lines.append(
            f"| `{name}` | {values['patients']} | {values['studies']} | "
            f"{per['min']}–{per['median']}–{per['max']} | {per['multi_study_patients']} |"
        )
    lines.append("")
    for name, values in manifests.items():
        if not values.get("splits"):
            continue
        lines += [f"### Split — `{name}`", "", "| Split | Patient | Study |", "|---|---:|---:|"]
        for split, counts in values["splits"].items():
            lines.append(f"| {split} | {counts['patients']} | {counts['studies']} |")
        lines.append("")
    leakage = cohort.get("leakage") or {}
    if leakage:
        lines += ["### Leakage check", "", "| Manifest | Patient nằm nhiều split | study_id trùng | Kết luận |", "|---|---:|---:|---|"]
        for name, values in leakage.items():
            verdict = "OK" if values["ok"] else "**FAIL**"
            lines.append(
                f"| `{name}` | {values['patients_crossing_splits']} | "
                f"{values['duplicate_study_id_count']} | {verdict} |"
            )
        lines.append("")
        offenders = {
            name: values["patients_crossing_splits_examples"]
            for name, values in leakage.items()
            if values["patients_crossing_splits_examples"]
        }
        if offenders:
            lines.append("Patient vi phạm (tối đa 20 mỗi manifest):")
            lines.append("")
            for name, examples in offenders.items():
                lines.append(f"- `{name}`: {', '.join(examples)}")
            lines.append("")
    membership = cohort.get("cohort_membership")
    if membership:
        lines += ["### Cohort membership", ""]
        for name, values in membership.items():
            counts = ", ".join(f"{key}={value}" for key, value in values["counts"].items())
            lines.append(f"- **{name}**: {counts}")
            for reason, count in list(values["exclusion_reasons"].items())[:5]:
                lines.append(f"    - loại vì `{reason}`: {count}")
        lines.append("")
    return lines


def _labels(summary: Mapping[str, Any]) -> list[str]:
    labels = summary.get("labels") or {}
    if labels.get("status") != "available":
        return ["## Nhãn diagnosis", "", f"Không phân tích được: {labels.get('reason', 'n/a')}", ""]
    lines = [
        "## Nhãn diagnosis", "",
        "Prevalence tính trên **dòng quan sát được**, không tính dòng missing — "
        "một nhãn thiếu không phải là nhãn âm.", "",
        "| Nhãn | Dương | Âm | Missing | Prevalence | Missing rate | Tỷ lệ mất cân bằng (âm:dương) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (labels.get("overall") or {}).items():
        lines.append(
            f"| `{name}` | {values['positive']} | {values['negative']} | {values['missing']} | "
            f"{_number(values['prevalence_of_observed'])} | {_number(values['missing_rate'])} | "
            f"{_number(values['imbalance_ratio'], 2)} |"
        )
    return [*lines, ""]


def _outcomes(summary: Mapping[str, Any]) -> list[str]:
    outcomes = summary.get("outcomes") or {}
    lines: list[str] = ["## Endpoint prognosis", ""]
    wrote = False
    for name, payload in outcomes.items():
        if not isinstance(payload, Mapping) or payload.get("status") != "available":
            continue
        wrote = True
        lines += [
            f"### `{name}` ({payload['rows']} dòng)", "",
            "| Endpoint | Event | Non-event | Missing | Event rate | Event ít nhất ở 1 split | Protocol gợi ý |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for endpoint, values in payload["endpoints"].items():
            lines.append(
                f"| `{endpoint}` | {values['events']} | {values['non_events']} | "
                f"{values['missing']} | {_number(values['event_rate_of_observed'])} | "
                f"{values['minimum_split_events']} | {values['recommended_protocol']} |"
            )
        lines.append("")
    if not wrote:
        lines.append("Không có manifest prognosis đọc được.")
        lines.append("")
    return lines


def _geometry(summary: Mapping[str, Any]) -> list[str]:
    geometry = summary.get("geometry") or {}
    lines = ["## Geometry và cache", ""]
    manifest = geometry.get("manifest_geometry") or {}
    if manifest.get("status") == "available":
        for column, axes in (manifest.get("axes") or {}).items():
            lines += [f"### `{column}`", "", "| Trục | n | min | p25 | median | p75 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
            for axis, values in axes.items():
                if not values:
                    continue
                lines.append(
                    f"| {axis} | {values['n']} | {_number(values['min'], 3)} | {_number(values['p25'], 3)} | "
                    f"{_number(values['median'], 3)} | {_number(values['p75'], 3)} | {_number(values['max'], 3)} |"
                )
            lines.append("")
    else:
        lines += [f"Manifest không có cột geometry: {manifest.get('reason', 'n/a')}", ""]
    cache = geometry.get("volume_cache") or {}
    if cache.get("status") == "available":
        lines += [
            f"Volume cache: {cache['files']} file, {cache['total_gib']:.2f} GiB tại `{cache['path']}`.",
            "",
        ]
    else:
        lines += [f"Volume cache: {cache.get('reason', 'n/a')}", ""]
    return lines


def _reports(summary: Mapping[str, Any]) -> list[str]:
    reports = summary.get("reports") or {}
    if reports.get("status") != "available":
        return ["## Report text", "", f"Không phân tích được: {reports.get('reason', 'n/a')}", ""]
    lines = ["## Report text", ""]
    text = reports.get("text") or {}
    if text.get("status") == "available":
        words = text.get("words") or {}
        lines += [
            f"Cột text: `{text['column']}` · {text['reports']} report · "
            f"rỗng {text['empty_reports']} ({_number(text['empty_rate'])})",
            "",
        ]
        if words:
            lines += [
                "| Đơn vị | min | p25 | median | p75 | max | mean |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| từ | {words['min']} | {words['p25']} | {words['median']} | "
                f"{words['p75']} | {words['max']} | {_number(words['mean'], 1)} |",
                "",
            ]
    coverage = reports.get("rule_coverage") or {}
    if coverage.get("status") == "available":
        lines += [
            f"### Rule extractor ({coverage['rule_version']}) giải quyết được bao nhiêu",
            "",
            f"Mẫu {coverage['reports_examined']} report. Trung bình "
            f"{_number(coverage['mean_targets_resolved_per_report'], 2)} / 19 target được regex "
            "giải quyết — phần còn lại phải trả bằng LLM hoặc thành abstention.",
            "",
            "| Target | Resolved | Tỷ lệ |", "|---|---:|---:|",
        ]
        ranked = sorted(
            coverage["by_target"].items(),
            key=lambda item: -(item[1]["resolved_rate"] or 0),
        )
        for target, values in ranked:
            lines.append(
                f"| `{target}` | {values['resolved']} | {_number(values['resolved_rate'])} |"
            )
        lines.append("")
    else:
        lines += [f"Rule coverage: {coverage.get('reason', 'n/a')}", ""]
    return lines


def _figures(summary: Mapping[str, Any]) -> list[str]:
    figures = summary.get("figures") or {}
    if not figures:
        return []
    lines = ["## Hình", "", "| Hình | Trạng thái | Đường dẫn / lý do |", "|---|---|---|"]
    for name, values in figures.items():
        detail = values.get("path") or values.get("reason") or "—"
        lines.append(f"| `{name}` | {values['status']} | `{detail}` |")
    return [*lines, ""]


def render(summary: Mapping[str, Any]) -> str:
    header = [
        f"# EDA — dataset profile `{summary['profile']}`",
        "",
        f"Chạy lúc {summary['generated_at']} · dataset root `{summary['dataset_root']}`",
        f"· output `{summary['output_dir']}`",
        "",
        "Báo cáo này chỉ **đọc** artifact dẫn xuất. Nó không sửa manifest và không tạo ra thứ gì",
        "training tiêu thụ, nên chạy lại lúc nào cũng được.",
        "",
    ]
    body: list[str] = []
    for section in (_inventory, _cohort, _labels, _outcomes, _geometry, _reports, _figures):
        body += section(summary)
    return "\n".join([*header, *body]).rstrip() + "\n"


def write(summary: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(summary), encoding="utf-8")
    return destination
