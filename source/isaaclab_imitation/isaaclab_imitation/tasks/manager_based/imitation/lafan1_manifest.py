from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Resolve the package root directly from this module to avoid importing the
# top-level package, which also registers Isaac tasks on import.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PACKAGE_ROOT / "manifests"
DEFAULT_G1_LAFAN1_MANIFEST_PATH = MANIFESTS_DIR / "g1_default_manifest.json"
DEFAULT_G1_DANCE102_MANIFEST_PATH = MANIFESTS_DIR / "g1_dance102_manifest.json"


def load_lafan1_manifest(manifest_path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load manifest entries and resolve relative motion paths against the manifest file."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"lafan1_manifest_path not found: {manifest_file}")

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries_like = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries_like is None:
            entries_like = data.get("lafan1_csv", data.get("motions", data))
    else:
        entries_like = data

    if not isinstance(entries_like, list) or len(entries_like) == 0:
        raise ValueError(
            "Manifest must define a non-empty `dataset.trajectories.lafan1_csv` list."
        )

    entries: list[dict[str, Any]] = []
    for index, entry_like in enumerate(entries_like):
        if not isinstance(entry_like, dict):
            raise ValueError(
                f"Manifest entry #{index} must be a mapping, got {type(entry_like)}."
            )

        path_value = entry_like.get("path") or entry_like.get("file")
        if path_value is None:
            raise ValueError(
                f"Manifest entry #{index} must include `path` (or `file`)."
            )
        if "input_fps" not in entry_like:
            raise ValueError(
                f"Manifest entry #{index} must include `input_fps`."
            )

        source_path = Path(str(path_value)).expanduser()
        if not source_path.is_absolute():
            source_path = (manifest_file.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()

        entries.append(
            {
                "name": str(entry_like.get("name") or source_path.stem),
                "path": str(source_path),
                "input_fps": float(entry_like["input_fps"]),
                **(
                    {"frame_range": entry_like["frame_range"]}
                    if "frame_range" in entry_like
                    else {}
                ),
            }
        )

    return manifest_file, entries


def dataset_path_from_entries(entries: list[dict[str, Any]]) -> str:
    """Create a stable cache path tied to the manifest entries."""
    signature = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/iltools_g1_lafan1_tracking_{digest}"
