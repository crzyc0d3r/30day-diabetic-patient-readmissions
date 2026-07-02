#!/usr/bin/env bash
# Local CI runner: executes the same Validate + E2E-smoke stages defined in
# azure-pipelines.yml / Jenkinsfile / cloudbuild.yaml, on the local host, so the
# pipeline definitions are runnable and reproducible without standing up a CI
# provider. The pipelines and this runner are described in README.md.
#
# Usage:
#   infra/ci-cd/run_ci_local.sh            # uses the active project env (`python`)
#   PYRUN="uv run --no-project --python 3.13 --with pytest --with ruff --with \
#     nbformat --with numpy --with pandas --with scipy --with statsmodels --" \
#     infra/ci-cd/run_ci_local.sh          # run in an ephemeral env
#
# Mirrors the upstream pipelines stage-for-stage; keep in lockstep when those change.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# PYRUN lets the caller front the python steps with a provisioned env (uv, venv).
PYRUN="${PYRUN:-}"
py() { ${PYRUN} python "$@"; }

fail=0
step() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
ok()   { printf '  [PASS] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }

step "Validate :: requirements pin lockstep (5 sites)"
py helpers/check_requirements_pins.py && ok "pins in lockstep" || bad "pin drift"

step "Validate :: ruff lint (advisory, matches CI advisory scope)"
${PYRUN} ruff check helpers/ test/ && ok "ruff clean" || echo "  (advisory - not gating)"

step "Validate :: structural pytest (no-model unit tests)"
py -m pytest test/test_cicd_trigger.py test/test_eda_stats.py test/test_constants.py \
    test/test_drift_sim.py -q && ok "unit tests pass" || bad "unit tests"

step "Validate :: every pipeline notebook parses as nbformat"
py -c "import nbformat, glob
fs = sorted(glob.glob('pipeline/*.ipynb'))
[nbformat.read(f, as_version=4) for f in fs]
print(f'  {len(fs)} notebooks parsed')" && ok "notebooks parse" || bad "notebook parse"

step "Validate :: remote-trigger handoff demo (mock CI server)"
py infra/ci-cd/demo_remote_trigger.py >/dev/null 2>&1 && ok "Airflow->CI trigger verified" || bad "trigger demo"

step "E2E smoke :: inference API /healthz on the running stack"
if curl -fsS -m 5 http://localhost:8002/healthz >/dev/null 2>&1; then
  ok "inference-api healthy ($(curl -fsS -m5 http://localhost:8002/healthz))"
else
  echo "  [skip] inference-api not reachable on :8002 (start the stack: mediwatch.sh init)"
fi

printf '\n'
if [ "$fail" -eq 0 ]; then printf '\033[32mCI LOCAL: PASS\033[0m\n'; else printf '\033[31mCI LOCAL: FAIL\033[0m\n'; fi
exit "$fail"
