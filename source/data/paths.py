from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PathConfigurationError(RuntimeError):
    pass


def discover_code_root(start: Path | None = None) -> Path:
    cursor = (start or Path(__file__)).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise PathConfigurationError(f"could not find pyproject.toml above {cursor}")


def _path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    text = os.path.expandvars(os.path.expanduser(str(value)))
    if "${" in text:
        raise PathConfigurationError(f"unresolved environment variable in path: {text}")
    return Path(text).resolve()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ProjectPaths:
    code_root: Path
    cloud_root: Path | None
    cloud_project_root: Path | None
    data_root: Path | None
    raw_inspect_root: Path | None
    derived_root: Path | None
    output_root: Path | None
    cache_root: Path
    third_party_root: Path

    @classmethod
    def resolve(
        cls,
        config: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        code_root: Path | None = None,
    ) -> ProjectPaths:
        configuration = dict((config or {}).get("paths") or {})
        environment = os.environ if env is None else env
        resolved_code = _path(configuration.get("project_code_root")) or (
            code_root or discover_code_root()
        ).resolve()
        cloud_root = _path(configuration.get("cloud_root") or environment.get("PE_CLOUD_ROOT"))
        cloud_project = _path(
            configuration.get("cloud_project_root") or environment.get("PE_CLOUD_PROJECT_ROOT")
        )
        if cloud_project is None and cloud_root is not None:
            cloud_project = (cloud_root / "pe-project").resolve()
        if cloud_project is None and resolved_code.name == "pe-project":
            cloud_project = resolved_code
        data_root = _path(configuration.get("data_root")) or (
            (cloud_root / "data").resolve() if cloud_root is not None else None
        )
        # PE_RAW_INSPECT_ROOT / PE_DERIVED_ROOT are deliberate overrides: paths.yaml can only
        # express a template under PE_CLOUD_ROOT, and the read-only release does not always
        # sit inside it. An explicitly exported override therefore wins over the template.
        raw = _path(
            environment.get("PE_RAW_INSPECT_ROOT") or configuration.get("raw_inspect")
        ) or ((data_root / "Stanford_INSPECT_dataset").resolve() if data_root is not None else None)
        derived = _path(
            environment.get("PE_DERIVED_ROOT") or configuration.get("derived_data")
        ) or ((data_root / "derived").resolve() if data_root is not None else None)
        output = _path(configuration.get("output_root")) or (
            (cloud_project / "outputs").resolve() if cloud_project is not None else None
        )
        cache = _path(
            configuration.get("cache_root") or environment.get("PE_LOCAL_CACHE_ROOT")
        ) or (resolved_code / "cache").resolve()
        third_party = _path(configuration.get("third_party_root")) or (
            resolved_code / "third_party"
        ).resolve()
        result = cls(
            code_root=resolved_code,
            cloud_root=cloud_root,
            cloud_project_root=cloud_project,
            data_root=data_root,
            raw_inspect_root=raw,
            derived_root=derived,
            output_root=output,
            cache_root=cache,
            third_party_root=third_party,
        )
        if output is not None:
            result.assert_persistent_output()
        return result

    def assert_persistent_output(self) -> Path:
        if self.cloud_project_root is None or self.output_root is None:
            raise PathConfigurationError(
                "persistent output is unavailable; set PE_CLOUD_ROOT or PE_CLOUD_PROJECT_ROOT"
            )
        expected = (self.cloud_project_root / "outputs").resolve()
        if self.output_root.resolve() != expected:
            raise PathConfigurationError(
                "OUTPUT_ROOT must be exactly cloud project outputs: "
                f"expected {expected}, got {self.output_root}"
            )
        return self.output_root

    def dataset_root(self, mode: str, profile: str | None = None) -> Path:
        """Root of the derived dataset the run reads from.

        ``mode`` stays ``full`` (full-data manifests, never a pilot code path). ``profile``
        selects one dataset profile built by ``source/data_preprocessing`` -- the cohort
        under ``derived/datasets/<profile>/``. Without a profile this is the historical
        flat derived root, so pre-profile manifests keep resolving unchanged.
        """
        if mode != "full":
            raise PathConfigurationError(f"unsupported data mode: {mode}")
        if self.derived_root is None:
            raise PathConfigurationError("full mode requires PE_CLOUD_ROOT/data/derived")
        name = str(profile or "").strip()
        return (self.derived_root / "datasets" / name) if name else self.derived_root

    def dataset_root_for(self, config: Mapping[str, Any] | None) -> Path:
        """Resolve the dataset root from a resolved config's ``data`` block."""
        data = dict((config or {}).get("data") or {})
        configured_root = data.get("root")
        if configured_root is not None and str(configured_root).strip():
            resolved_root = _path(configured_root)
            if resolved_root is not None:
                return resolved_root
        return self.dataset_root(str(data.get("mode") or "full"), data.get("profile"))

    def code_asset(self, value: Any) -> Path | None:
        """Resolve a repository-local model/config asset independently of the shell CWD."""
        if value is None or str(value).strip() == "":
            return None
        text = os.path.expandvars(os.path.expanduser(str(value)))
        if "${" in text:
            raise PathConfigurationError(f"unresolved environment variable in path: {text}")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.code_root / candidate
        return candidate.resolve()

    def output_asset(self, value: Any) -> Path | None:
        """Resolve an explicitly configured artifact while preventing output-root escape."""
        if value is None or str(value).strip() == "":
            return None
        text = os.path.expandvars(os.path.expanduser(str(value)))
        if "${" in text:
            raise PathConfigurationError(f"unresolved environment variable in path: {text}")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.assert_persistent_output() / candidate
        return self.require_within_output(candidate)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "project_code_root": str(self.code_root),
            "cloud_root": str(self.cloud_root) if self.cloud_root else None,
            "cloud_project_root": str(self.cloud_project_root) if self.cloud_project_root else None,
            "data_root": str(self.data_root) if self.data_root else None,
            "raw_inspect": str(self.raw_inspect_root) if self.raw_inspect_root else None,
            "derived_data": str(self.derived_root) if self.derived_root else None,
            "output_root": str(self.output_root) if self.output_root else None,
            "cache_root": str(self.cache_root),
            "third_party_root": str(self.third_party_root),
        }

    def require_within_output(self, candidate: Path) -> Path:
        root = self.assert_persistent_output().resolve()
        resolved = candidate.resolve()
        if not _inside(resolved, root):
            raise PathConfigurationError(f"path escapes persistent output root: {candidate}")
        return resolved
