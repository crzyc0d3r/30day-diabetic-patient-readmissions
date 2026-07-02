#!/usr/bin/env python3
"""Diff the four lockstep pin sites and report drift, with PEP 440 normalisation.

Drift vs. local-version flavor (PEP 440)
----------------------------------------
`torch==2.11.0` (top-level CPU/generic wheel) and `torch==2.11.0+cu130`
(ray GPU image) are the same upstream release expressed for two different
build flavors. The `+cu130` suffix is a PEP 440 local version segment.
This script normalises that segment away before comparing, so the
inventory across the four pin sites can carry whichever wheel flavor each
target environment actually installs without showing up as drift.

What counts as drift
--------------------
* Different upstream versions for the same package across two or more
  pin sites (e.g. `scikit-learn==1.8.0` vs `scikit-learn==1.7.0`).
* Missing pin sites (all five files must exist).

What doesn't count as drift
---------------------------
* Different local-version segments (`+cu130` vs `+cpu` vs unsuffixed)
  on the same upstream release. These are intentionally diverse so a
  CPU laptop, an aarch64 box, and a CUDA-13 host can each install the
  right wheel.
* A package present in one pin site but not the others. Drift is only
  reported when the SAME package disagrees on the upstream version
  across two or more sites.

`requirements.txt`'s header documents that five pin sites must move together:

  * `requirements.txt` (notebook plus DAG worker)
  * `infra/inference-api/requirements.txt` (FastAPI inference container)
  * `infra/ray/Dockerfile` (Ray HPO worker image)
  * `infra/airflow/requirements.txt` (Airflow base interpreter / retrain-DAG
    worker)
  * `infra/airflow/ray-driver-requirements.txt` (the isolated CPython 3.13.2
    venv the retrain DAG's HPO task runs under, so the Ray *driver* sits on the
    same cloudpickle boundary as the Ray worker image and must match it)

This script reads the five sites, extracts `package==version` pins from
each, and prints a table of any package that appears in at least two
sites but with mismatched versions.

Exit status:
    0 if no drift detected
    1 if any common package disagrees across sites

Designed to be lightweight (stdlib only) so it can run in the validate
stage of azure-pipelines.yml without extra installations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SITES = {
    "top-level":     ROOT / "requirements.txt",
    "inference-api": ROOT / "infra" / "inference-api" / "requirements.txt",
    "ray-worker":    ROOT / "infra" / "ray" / "Dockerfile",
    "airflow":       ROOT / "infra" / "airflow" / "requirements.txt",
    # The isolated CPython 3.13.2 Ray-driver venv baked into the Airflow image
    # (infra/airflow/Dockerfile). Its libraries sit on the same cloudpickle
    # boundary as the ray-worker image, so they must move in lockstep too.
    "ray-driver":    ROOT / "infra" / "airflow" / "ray-driver-requirements.txt",
}

PIN_RE = re.compile(r"([A-Za-z0-9_.\-\[\]]+)==([A-Za-z0-9_.\-+]+)")


def normalise(pkg: str) -> str:
    """Strip extras like `uvicorn[standard]` and lowercase for matching."""
    base = pkg.split("[", 1)[0]
    return base.lower().replace("_", "-")


def upstream_version(ver: str) -> str:
    """Drop the PEP 440 local-version segment (`+cu130`, `+cpu`, etc.).

    `torch==2.11.0` (top-level CPU/generic wheel) and `torch==2.11.0+cu130`
    (ray GPU image) are the same upstream release, just different build
    flavors. The drift check treats them as matching. The display
    table still shows the originals so the per-site flavor is visible.
    """
    return ver.split("+", 1)[0]


def parse_pins(path: Path) -> dict[str, str]:
    """Return {normalised_package: version_string} for every `pkg==ver` in path."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        for match in PIN_RE.finditer(stripped):
            pkg, ver = match.group(1), match.group(2)
            out[normalise(pkg)] = ver
    return out


def main() -> int:
    parsed = {label: parse_pins(p) for label, p in SITES.items()}

    missing = [label for label, p in SITES.items() if not p.exists()]
    if missing:
        print(f"ERROR: missing pin sites: {', '.join(missing)}", file=sys.stderr)
        return 1

    all_pkgs = sorted({pkg for pins in parsed.values() for pkg in pins})

    drift: list[tuple[str, dict[str, str]]] = []
    for pkg in all_pkgs:
        seen = {label: pins.get(pkg, "-") for label, pins in parsed.items()}
        # Drift = pkg appears in 2+ sites, AND the upstream versions
        # disagree. Local-version segments ('+cu130', '+cpu') are stripped
        # before comparison. See 'upstream_version' for why.
        present = [v for v in seen.values() if v != "-"]
        if len(present) >= 2 and len({upstream_version(v) for v in present}) > 1:
            drift.append((pkg, seen))

    label_widths = [max(len(label), 12) for label in SITES]
    header = f"{'package':<24}" + "".join(f"  {label:<{w}}" for label, w in zip(SITES, label_widths))
    print(header)
    print("-" * len(header))

    if not drift:
        print(f"(no drift detected across the {len(SITES)} pin sites)")
        return 0

    for pkg, seen in drift:
        row = f"{pkg:<24}" + "".join(f"  {seen[label]:<{w}}" for label, w in zip(SITES, label_widths))
        print(row)

    print()
    print(f"DRIFT: {len(drift)} package(s) disagree across pin sites.")
    print("Update all four sites in the same change. See requirements.txt header.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
