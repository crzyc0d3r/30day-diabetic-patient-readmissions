"""Unit tests for helpers/check_requirements_pins.py.

WHAT this module guards.
`check_requirements_pins` is the stdlib-only CI guard that keeps the four
"lockstep" pin sites (top-level requirements, the inference-api container,
the Ray worker Dockerfile, and the Airflow worker requirements) from
drifting apart. It exists because `torch==2.11.0` and `torch==2.11.0+cu130`
are the SAME upstream release built for two wheel flavors, so they must NOT
register as drift, while two genuinely different upstream versions
(`1.7.0` vs `1.8.0`) MUST.

WHY these tests.
The script has no other test coverage and runs in a pipeline validate stage
where a silent regression (for example, comparing local segments or failing
to normalise extras) would either let real drift through or spam false
positives. Each test below pins one documented behaviour so an edit that
breaks it is caught immediately. We never touch the real requirements files:
every parse target is written under `tmp_path`, and `main` is exercised
by monkeypatching `crp.SITES` to point at temp files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import check_requirements_pins as crp


# normalise(pkg)
#
# WHAT: lowercases, turns underscores into hyphens, and strips PEP 508 extras
#       such as `uvicorn[standard]` down to the bare distribution name.
# WHY:  the four sites spell the same package differently (extras on one,
#       casing on another). Matching must happen on a canonical key, or the
#       drift comparison silently misses packages that are the same.
@pytest.mark.parametrize(
    "raw, expected",
    [
        # Extras are dropped so the inference-api's `uvicorn[standard]` lines
        # up with a bare `uvicorn` pin elsewhere.
        ("uvicorn[standard]", "uvicorn"),
        # Casing is normalised so `Flask` and `flask` collide on one key.
        ("Flask", "flask"),
        # Underscores become hyphens (PEP 503 style) so the two spellings of
        # the same project match.
        ("scikit_learn", "scikit-learn"),
        # Combined: extras, mixed case, and an underscore at once.
        ("My_Pkg[Extra]", "my-pkg"),
        # An already-canonical name passes through untouched.
        ("scikit-learn", "scikit-learn"),
    ],
)
def test_normalise_canonicalises_package_names(raw: str, expected: str) -> None:
    """normalise must strip extras, lowercase, and hyphenate underscores."""
    assert crp.normalise(raw) == expected


# upstream_version(ver)
#
# WHAT: drops the PEP 440 local-version segment (everything after the first
#       `+`) so build-flavor suffixes do not count as drift.
# WHY:  `2.11.0+cu130` (CUDA wheel) and `2.11.0` (generic wheel) are the same
#       release. The drift check compares upstream versions, so this is the
#       function that makes flavor diversity legal.
@pytest.mark.parametrize(
    "ver, expected",
    [
        # The headline case from the module docstring: CUDA local segment
        # stripped back to the shared upstream release.
        ("2.11.0+cu130", "2.11.0"),
        # A CPU flavor strips to the same upstream as the CUDA one above.
        ("2.11.0+cpu", "2.11.0"),
        # No local segment means the version returns unchanged.
        ("1.7.0", "1.7.0"),
        # Only the FIRST `+` matters. Anything after it stays dropped.
        ("3.0.0+local+weird", "3.0.0"),
    ],
)
def test_upstream_version_strips_local_segment(ver: str, expected: str) -> None:
    """upstream_version must remove the PEP 440 local segment after `+`."""
    assert crp.upstream_version(ver) == expected


# parse_pins(path)
#
# WHAT: reads a requirements-style file and returns
#       {normalised_package: version_string} for every `pkg==ver` line,
#       skipping comments and blanks.
# WHY:  this is the per-site extraction the whole drift report is built on. It
#       must normalise keys (so sites match) yet preserve the raw version
#       string (so the display table can still show `+cu130` per site), and a
#       missing file must degrade to an empty dict rather than raising.
def test_parse_pins_extracts_and_normalises(tmp_path: Path) -> None:
    """parse_pins normalises keys, keeps raw versions, and skips noise lines."""
    req = tmp_path / "requirements.txt"
    # A realistic mix: a header comment, a blank line, an extras spec, a
    # local-version (+cu130) pin, mixed casing, and an inline-comment line.
    req.write_text(
        "\n".join(
            [
                "# lockstep pin site - keep in sync with the other three",
                "",
                "uvicorn[standard]==0.30.0",
                "torch==2.11.0+cu130",
                "Scikit_Learn==1.8.0",
                "numpy==2.1.0",
            ]
        ),
        encoding="utf-8",
    )

    pins = crp.parse_pins(req)

    # Keys are canonicalised: extras stripped, lowercased, underscores
    # hyphenated. Versions stay verbatim (the +cu130 remains) so the display
    # table can show the per-site flavor.
    assert pins == {
        "uvicorn": "0.30.0",
        "torch": "2.11.0+cu130",
        "scikit-learn": "1.8.0",
        "numpy": "2.1.0",
    }


def test_parse_pins_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    """Comment and blank lines must contribute nothing to the pin dict."""
    req = tmp_path / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "# this whole line is a comment with a fake==1.0.0 pin",
                "   ",
                "",
                "realpkg==9.9.9",
            ]
        ),
        encoding="utf-8",
    )

    pins = crp.parse_pins(req)

    # The commented `fake==1.0.0` must be ignored. Only the real pin survives.
    assert pins == {"realpkg": "9.9.9"}
    assert "fake" not in pins


def test_parse_pins_missing_path_returns_empty_dict(tmp_path: Path) -> None:
    """A non-existent path returns {} instead of raising (lenient by design)."""
    missing = tmp_path / "does_not_exist.txt"
    assert crp.parse_pins(missing) == {}


# main()
#
# WHAT: reads the four SITES paths, prints a drift table, and returns 0 (no
#       drift) or 1 (drift detected, or a site file missing).
# WHY:  this is the exit code the CI validate stage gates on. We monkeypatch
#       crp.SITES onto temp files so we never read the real lock files and can
#       construct each scenario precisely.
def _make_sites(tmp_path: Path, monkeypatch, contents: dict[str, str | None]) -> None:
    """Write each site's contents under tmp_path and point crp.SITES at them.

    `contents` maps a site label to its file body, or to `None` to mean
    "this site file is intentionally absent" (so we can exercise the missing
    -site branch). The path is still registered in SITES so `main` sees it
    and reports it as missing.
    """
    sites: dict[str, Path] = {}
    for label, body in contents.items():
        path = tmp_path / f"{label}.txt"
        if body is not None:
            path.write_text(body, encoding="utf-8")
        sites[label] = path
    monkeypatch.setattr(crp, "SITES", sites)


def test_main_returns_zero_when_upstream_agrees(tmp_path: Path, monkeypatch, capsys) -> None:
    """All four sites agree on the upstream version -> no drift -> exit 0.

    The torch pins differ only by local segment (+cu130, +cpu, unsuffixed),
    which the module treats as NON-drift, so main must return 0 and print the
    'no drift' banner.
    """
    _make_sites(
        tmp_path,
        monkeypatch,
        {
            "top-level": "torch==2.11.0\nnumpy==2.1.0\n",
            "inference-api": "torch==2.11.0+cpu\nnumpy==2.1.0\n",
            "ray-worker": "torch==2.11.0+cu130\nnumpy==2.1.0\n",
            "airflow": "torch==2.11.0+cu130\nnumpy==2.1.0\n",
        },
    )

    rc = crp.main()

    assert rc == 0
    out = capsys.readouterr().out
    # The "no drift" message must appear when versions line up.
    assert "no drift detected" in out


def test_main_returns_one_on_upstream_disagreement(tmp_path: Path, monkeypatch, capsys) -> None:
    """Two sites disagree on the UPSTREAM version -> drift -> exit 1.

    scikit-learn is pinned to 1.8.0 on two sites and 1.7.0 on two others.
    These are different upstream releases, not just flavors, so this is real
    drift: main must return 1 and the package must appear in the table.
    """
    _make_sites(
        tmp_path,
        monkeypatch,
        {
            "top-level": "scikit-learn==1.8.0\n",
            "inference-api": "scikit-learn==1.8.0\n",
            "ray-worker": "scikit-learn==1.7.0\n",
            "airflow": "scikit-learn==1.7.0\n",
        },
    )

    rc = crp.main()

    assert rc == 1
    out = capsys.readouterr().out
    # The drift summary and the offending package must both be shown.
    assert "DRIFT:" in out
    assert "scikit-learn" in out


def test_main_returns_one_when_a_site_is_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    """A missing pin-site file -> exit 1, even when present sites agree.

    The contract is "all four files must exist". We give three valid sites
    and leave the fourth absent. main must short-circuit to 1 and name the
    missing site on stderr.
    """
    _make_sites(
        tmp_path,
        monkeypatch,
        {
            "top-level": "numpy==2.1.0\n",
            "inference-api": "numpy==2.1.0\n",
            "ray-worker": "numpy==2.1.0\n",
            # Fourth site file is intentionally not written to disk.
            "airflow": None,
        },
    )

    rc = crp.main()

    assert rc == 1
    err = capsys.readouterr().err
    # The missing-site error names which label could not be found.
    assert "missing pin sites" in err
    assert "airflow" in err
