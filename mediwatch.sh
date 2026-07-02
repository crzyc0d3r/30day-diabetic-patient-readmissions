#!/usr/bin/env bash
# mediwatch.sh - command-line interface for the medi-watch platform.
#
# USAGE
# -----
#   ./mediwatch.sh <command> [options]
#
# COMMANDS
# --------
#   doctor    Full audit + optional autofix. Use --no-fix for a read-only report.
#   init      Bring the FULL stack up (postgres, airflow, mlflow, ray, prom, grafana,
#             inference-api, driftly). Preflight audit + optional --reset.
#   install   Register a 'medi-watch' shim on PATH so commands work from any directory.
#   train     Run the DATA-PREP notebooks (01..05) on the host. Aliased as 'run'.
#   retrain   Trigger the airflow 'retrain_on_drift' DAG (HPO+train+register, NB06..08)
#             entirely in containers. The host only fires the trigger.
#   drift     Stage a simulated production batch + trigger the 'scheduled_drift_check'
#             DAG (auto-triggers retrain on an ALERT verdict).
#   activate  Serve-only: bring up ONLY postgres + mlflow + inference-api against an
#             already-trained champion. Lean alternative to 'init' for hosting
#             predictions.
#   k8s       Apply infra/k8s manifests (mlflow → ray → airflow → inference-api) to
#             the current kubectl context. Aliased as 'k8'.
#   shutdown  Halt containers (--reset wipes volumes, ALL project images, and data/).
#   help      Show this help.
#
# Run './mediwatch.sh <command> --help' for command-specific help.
#
# EXAMPLES
# --------
#   ./mediwatch.sh doctor               # audit + interactive autofix
#   ./mediwatch.sh doctor --no-fix      # read-only audit report only (no prompts)
#   ./mediwatch.sh doctor --yes         # audit + auto-accept every fixable finding
#   ./mediwatch.sh init --no-fix        # skip the autofix prompt
#   ./mediwatch.sh install              # sudo install to /usr/local/bin/medi-watch
#   ./mediwatch.sh install --user       # no-sudo install to ~/.local/bin/medi-watch
#   ./mediwatch.sh train                # data prep 01..05 on the host
#   ./mediwatch.sh retrain              # trigger airflow HPO->train->register (NB06..08)
#   ./mediwatch.sh drift coding_shift   # stage a drift batch + trigger the drift-check DAG
#   ./mediwatch.sh activate             # serve-only: host the existing champion
#   ./mediwatch.sh k8s                  # apply k8s manifests to current context
#   ./mediwatch.sh k8s --build --reset  # build+load images, wipe ns, then apply
#   ./mediwatch.sh shutdown --reset --yes

# === mediwatch CLI ===
set -euo pipefail

# ---------- repo paths ----------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$ROOT/infra"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

# ---------- platform detection ----------
# OS_KIND is one of: linux | macos | windows | unknown. Used to gate the
# Linux-only GPU probes, choose the right package-manager hint (apt vs brew),
# and pick portable temp paths. Native Windows users should run mediwatch.ps1
# instead. Under WSL2 uname reports Linux and this script runs as on Ubuntu.
case "$(uname -s 2>/dev/null)" in
  Linux*)               OS_KIND="linux"   ;;
  Darwin*)              OS_KIND="macos"   ;;
  MINGW*|MSYS*|CYGWIN*) OS_KIND="windows" ;;
  *)                    OS_KIND="unknown" ;;
esac

# Portable temp dir. macOS sets TMPDIR (a per-user path), Linux usually does
# not, so fall back to /tmp. Holds the pip-freeze drift cache and curl
# scratch files. Trailing slash stripped so "$TMP_DIR/foo" never doubles it.
TMP_DIR="${TMPDIR:-/tmp}"; TMP_DIR="${TMP_DIR%/}"

# Bash 4+ guard. This script uses `mapfile` and other bash-4 features. macOS
# ships bash 3.2, so fail early with an actionable hint rather than at the first
# `mapfile` call deep inside a teardown.
if [ -z "${BASH_VERSINFO:-}" ] || [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
  printf 'mediwatch.sh requires bash 4 or newer (found %s).\n' "${BASH_VERSION:-unknown}" >&2
  if [ "$OS_KIND" = "macos" ]; then
    printf 'macOS ships bash 3.2. Install a modern bash and re-run, e.g.:\n' >&2
    printf '  brew install bash && exec /opt/homebrew/bin/bash %s "$@"\n' "$0" >&2
    printf '  (Intel Macs: /usr/local/bin/bash)\n' >&2
  fi
  exit 1
fi

# ---------- GPU-aware compose file selection ----------
# The base infra/docker-compose.yml is deliberately GPU-free so it boots on any
# host. infra/docker-compose.gpu.yml is an additive overlay that re-attaches the
# NVIDIA device reservations for ray-head and ray-worker. We include it
# only when a usable GPU is present, by exporting COMPOSE_FILE, which every
# `docker compose` invocation in this script (all of which `cd "$INFRA"` first)
# picks up automatically, so no individual call site needs to change.
#
# "Usable GPU" == Linux host where `nvidia-smi` succeeds. On macOS/Windows/CPU
# Linux the overlay is omitted and the stack runs CPU-only.
_have_usable_gpu() {
  [ "$OS_KIND" = "linux" ] || return 1
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1
}

# select_compose_files: set + export COMPOSE_FILE for the current host.
# Idempotent: safe to call from any lifecycle command before it shells out to
# docker compose.
# COMPOSE_FILE uses ':'-separated paths relative to $INFRA (the compose cwd).
select_compose_files() {
  if _have_usable_gpu; then
    export COMPOSE_FILE="docker-compose.yml:docker-compose.gpu.yml"
  else
    export COMPOSE_FILE="docker-compose.yml"
  fi
}

# pkg_hint <docker|compose|git|python>: per-OS install instruction shown
# in audit findings and the autofix "manual" list. On Linux we keep the apt
# recipe (which the autofix can run). macOS/Windows get brew/winget guidance,
# since on those hosts we degrade to manual hints rather than auto-mutating
# the system.
pkg_hint() {
  case "$OS_KIND-$1" in
    linux-docker)   echo "install via get.docker.com" ;;
    linux-compose)  echo "apt install docker-compose-plugin" ;;
    linux-git)      echo "apt install git" ;;
    linux-python)   echo "install Python 3.13.x and activate an environment" ;;
    macos-docker)   echo "install Docker Desktop: brew install --cask docker" ;;
    macos-compose)  echo "bundled with Docker Desktop (enable Compose v2 in settings)" ;;
    macos-git)      echo "brew install git  (or: xcode-select --install)" ;;
    macos-python)   echo "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
    *-docker)       echo "install Docker Desktop for your platform" ;;
    *-compose)      echo "bundled with Docker Desktop" ;;
    *-git)          echo "install git from https://git-scm.com" ;;
    *-python)       echo "install Python 3.13+" ;;
    *)              echo "install $1" ;;
  esac
}

# ---------- version (shown in banner) ----------
VERSION="1.0.900"

# ============================================================================
# pretty output: colors, badges, banner
# ============================================================================

C_RESET="" C_BOLD="" C_DIM=""
C_CYAN="" C_YELLOW="" C_MAGENTA="" C_GREEN="" C_AMBER="" C_RED="" C_INK2=""
BG_GREEN="" BG_AMBER="" BG_RED="" BG_BRED="" BG_GRAY=""
FG_BLACK="" FG_WHITE=""
IS_TTY=0
HAS_TRUECOLOR=0

init_colors() {
  [ -t 1 ] && IS_TTY=1 || IS_TTY=0
  if [ "${NO_COLOR:-}" = "1" ] || [ "$IS_TTY" -eq 0 ]; then
    return
  fi
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_CYAN=$'\033[38;5;81m'
  C_YELLOW=$'\033[38;5;227m'
  C_MAGENTA=$'\033[38;5;207m'
  C_GREEN=$'\033[38;5;83m'
  C_AMBER=$'\033[38;5;214m'
  C_RED=$'\033[38;5;203m'
  C_INK2=$'\033[38;5;249m'
  BG_GREEN=$'\033[48;5;83m'
  BG_AMBER=$'\033[48;5;214m'
  BG_RED=$'\033[48;5;203m'
  BG_BRED=$'\033[48;5;196m'
  BG_GRAY=$'\033[48;5;240m'
  FG_BLACK=$'\033[38;5;232m'
  FG_WHITE=$'\033[38;5;255m'
  case "${COLORTERM:-}" in
    truecolor|24bit) HAS_TRUECOLOR=1 ;;
  esac
}
init_colors

badge_ok()   { printf '%s%s OK %s'   "$BG_GREEN" "$FG_BLACK" "$C_RESET"; }
badge_warn() { printf '%s%sWARN%s'   "$BG_AMBER" "$FG_BLACK" "$C_RESET"; }
badge_miss() { printf '%s%sMISS%s'   "$BG_RED"   "$FG_BLACK" "$C_RESET"; }
badge_stop() { printf '%s%sSTOP%s'   "$BG_GRAY"  "$FG_WHITE" "$C_RESET"; }
badge_skip() { printf '%s%sSKIP%s'   "$BG_GRAY"  "$FG_WHITE" "$C_RESET"; }
badge_fail() { printf '%s%s%sFAIL%s' "$BG_BRED"  "$FG_WHITE" "$C_BOLD"  "$C_RESET"; }

# print a gradient cyan->magenta rule. Falls back to solid cyan on non-truecolor.
gradient_rule() {
  local w=${1:-63}
  if [ "$HAS_TRUECOLOR" -eq 1 ] && [ "$IS_TTY" -eq 1 ]; then
    local i r g b
    for (( i=0; i<w; i++ )); do
      r=$(( 95 + (255 - 95) * i / (w - 1) ))
      g=$(( 215 + (95 - 215) * i / (w - 1) ))
      b=255
      printf '\033[38;2;%d;%d;%dm━' "$r" "$g" "$b"
    done
    printf '%s\n' "$C_RESET"
  else
    local line=""
    for (( i=0; i<w; i++ )); do line="${line}━"; done
    printf '%s%s%s\n' "$C_CYAN" "$line" "$C_RESET"
  fi
}

print_banner() {
  local subtitle="${1:-mediwatch-cli}"
  echo
  gradient_rule 63
  printf '  %sMEDI-WATCH%s  %s·%s  %-22s %sv%s%s\n' \
    "$C_BOLD" "$C_RESET" "$C_DIM" "$C_RESET" "$subtitle" "$C_DIM" "$VERSION" "$C_RESET"
  gradient_rule 63
  echo
}

hr_section() {
  # hr_section <color-var> <title>
  local color="$1" title="$2"
  printf '\n  %s━━━━ %s ━━━━%s\n' "$color" "$title" "$C_RESET"
}

# Per-state row counters. Reset at the top of each run_audit, incremented by
# row() so the tally stays exact regardless of which probes are silent.
OK_COUNT=0
WARN_COUNT=0
MISS_COUNT=0
STOP_COUNT=0
SKIP_COUNT=0

# Print a finding row.
# usage: row <badge_func> <item-name> <detail>
row() {
  local badge_func="$1" name="$2" detail="${3:-}"
  case "$badge_func" in
    badge_ok)   OK_COUNT=$((OK_COUNT+1)) ;;
    badge_warn) WARN_COUNT=$((WARN_COUNT+1)) ;;
    badge_miss) MISS_COUNT=$((MISS_COUNT+1)) ;;
    badge_stop) STOP_COUNT=$((STOP_COUNT+1)) ;;
    badge_skip) SKIP_COUNT=$((SKIP_COUNT+1)) ;;
  esac
  printf '  %s  %-26s %s%s%s\n' \
    "$($badge_func)" "$name" "$C_INK2" "$detail" "$C_RESET"
}

note() { printf '       %s» %s%s\n' "$C_DIM" "$1" "$C_RESET"; }
warn() { printf '  %s[warn]%s %s\n' "$C_AMBER" "$C_RESET" "$1"; }
info() { printf '  %s\n' "$1"; }
die()  { printf '  %s[fail]%s %s\n' "$C_RED"   "$C_RESET" "$1" >&2; exit 1; }

# ============================================================================
# probes: read-only. each returns 0=ok, 1=miss, 2=outdated, 3=unhealthy.
# findings array accumulates "STATE\tNAME\tDETAIL\tCATEGORY" lines.
# ============================================================================

FINDINGS=()
add_finding() {
  # STATE NAME DETAIL CATEGORY
  FINDINGS+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4")
}

# tiny semver compare: 0 if A>=B, 1 otherwise. only major.minor used.
ver_ge() {
  local a="$1" b="$2"
  local a_maj=${a%%.*}; local a_rest=${a#*.}; local a_min=${a_rest%%.*}
  local b_maj=${b%%.*}; local b_rest=${b#*.}; local b_min=${b_rest%%.*}
  a_maj=${a_maj:-0}; a_min=${a_min:-0}; b_maj=${b_maj:-0}; b_min=${b_min:-0}
  if [ "$a_maj" -gt "$b_maj" ]; then return 0; fi
  if [ "$a_maj" -lt "$b_maj" ]; then return 1; fi
  [ "$a_min" -ge "$b_min" ]
}

probe_docker() {
  local name="Docker Engine"
  set +e
  command -v docker >/dev/null 2>&1
  local has=$?
  set -e
  if [ "$has" -ne 0 ]; then
    row badge_miss "$name" "not found"
    add_finding "miss" "$name" "$(pkg_hint docker)" "host_docker"
    return 1
  fi
  local v
  v=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',') || v="unknown"
  if ver_ge "$v" "24.0"; then
    row badge_ok "$name" "$v"
    return 0
  else
    row badge_warn "$name" "$v (need ≥24.0)"
    add_finding "warn" "$name" "version $v predates 24.0 cutoff" "host_docker"
    return 2
  fi
}

probe_compose() {
  local name="Docker Compose v2"
  set +e
  docker compose version >/dev/null 2>&1
  local has=$?
  set -e
  if [ "$has" -ne 0 ]; then
    row badge_miss "$name" "compose plugin not found"
    add_finding "miss" "$name" "$(pkg_hint compose)" "host_compose"
    return 1
  fi
  local v
  v=$(docker compose version --short 2>/dev/null) || v="unknown"
  if ver_ge "$v" "2.20"; then
    row badge_ok "$name" "$v"
    return 0
  else
    row badge_warn "$name" "$v (need ≥2.20)"
    add_finding "warn" "$name" "compose plugin predates 2.20" "host_compose"
    return 2
  fi
}

probe_nvidia_driver() {
  local name="NVIDIA driver"
  if [ "$OS_KIND" != "linux" ]; then
    row badge_skip "$name" "GPU stack is Linux-only (skipped on $OS_KIND)"
    return 0
  fi
  set +e
  nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1
  local has=$?
  set -e
  if [ "$has" -ne 0 ]; then
    row badge_miss "$name" "nvidia-smi not available"
    add_finding "miss" "$name" "ubuntu-drivers autoinstall (REBOOT required)" "host_nvdrv"
    return 1
  fi
  local v
  v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  row badge_ok "$name" "$v"
  return 0
}

probe_cuda() {
  local name="CUDA"
  if [ "$OS_KIND" != "linux" ]; then
    row badge_skip "$name" "GPU stack is Linux-only (skipped on $OS_KIND)"
    return 0
  fi
  set +e
  local cuda_v
  cuda_v=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}')
  set -e
  if [ -z "$cuda_v" ]; then
    row badge_miss "$name" "no CUDA reported by nvidia-smi"
    add_finding "miss" "$name" "apt install cuda-toolkit" "host_cuda"
    return 1
  fi
  if ver_ge "$cuda_v" "12.0"; then
    row badge_ok "$name" "$cuda_v"
    return 0
  else
    row badge_warn "$name" "$cuda_v (need 12.x or 13.x)"
    add_finding "warn" "$name" "CUDA $cuda_v is older than supported" "host_cuda"
    return 2
  fi
}

probe_nvidia_ctk() {
  local name="nvidia-container-tk"
  if [ "$OS_KIND" != "linux" ]; then
    row badge_skip "$name" "Docker Desktop GPU runtime (skipped on $OS_KIND)"
    return 0
  fi
  set +e
  command -v nvidia-ctk >/dev/null 2>&1
  local has=$?
  set -e
  if [ "$has" -ne 0 ]; then
    row badge_miss "$name" "nvidia-ctk not on PATH"
    add_finding "miss" "$name" "apt install nvidia-container-toolkit + nvidia-ctk runtime configure" "host_nvctk"
    return 1
  fi
  # check daemon.json has the nvidia runtime registered
  if [ -r /etc/docker/daemon.json ] && grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
    local v
    v=$(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}') || v="installed"
    row badge_ok "$name" "$v"
    return 0
  else
    row badge_warn "$name" "installed but docker runtime not registered"
    add_finding "warn" "$name" "run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" "host_nvctk"
    return 3
  fi
}

probe_python() {
  local name="python"
  set +e
  command -v python >/dev/null 2>&1
  local has=$?
  set -e
  if [ "$has" -ne 0 ]; then
    row badge_miss "$name" "not found"
    add_finding "miss" "$name" "$(pkg_hint python)" "host_python"
    return 1
  fi
  local v
  v=$(python -V 2>&1 | awk '{print $2}') || v="unknown"
  if ver_ge "$v" "3.13.0"; then
    row badge_ok "$name" "$v"
    return 0
  else
    row badge_warn "$name" "$v (need ≥3.13)"
    add_finding "warn" "$name" "python $v is older than 3.13" "host_python"
    return 2
  fi
}

probe_git() {
  local name="git"
  if command -v git >/dev/null 2>&1; then
    row badge_ok "$name" "$(git --version 2>/dev/null | awk '{print $3}')"
    return 0
  fi
  row badge_miss "$name" "not found"
  add_finding "miss" "$name" "$(pkg_hint git)" "host_git"
  return 1
}

probe_bash() {
  local name="bash"
  row badge_ok "$name" "${BASH_VERSION:-running}"
  return 0
}

# probe_python_libs: verify the libraries pinned in requirements.txt are
# installed in whatever Python is active. The environment name (conda, venv,
# or bare) does not matter: the check confirms only that every dependency
# listed in requirements.txt is installed in the interpreter on PATH.
probe_python_libs() {
  local name="requirements.txt"
  if ! command -v python >/dev/null 2>&1; then
    row badge_miss "$name" "python not on PATH (activate your env)"
    add_finding "miss" "$name" "activate a Python 3.13 env, then re-audit" "py_medi_watch"
    return 1
  fi
  local pyv
  pyv=$(python -V 2>&1 | awk '{print $2}') || pyv="?"

  local req="$ROOT/requirements.txt"
  if [ ! -r "$req" ]; then
    row badge_ok "$name" "Python $pyv (no requirements.txt to check)"
    return 0
  fi

  # Snapshot installed packages via `pip freeze`, cached and keyed by active env
  # so switching envs doesn't reuse a stale freeze from a different interpreter.
  # Refresh whenever requirements.txt is newer than the cached freeze.
  local env_slug="${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-noenv}}"
  env_slug="${env_slug//\//_}"
  local cache="$TMP_DIR/medi-watch-audit-reqs.${env_slug}.lock"
  if [ ! -r "$cache" ] || [ "$req" -nt "$cache" ]; then
    # Enumerate installed distributions WITHOUT shelling out to pip: uv-created
    # venvs ship no `pip` module, so `python -m pip freeze` prints nothing and
    # every dependency would look "missing". importlib.metadata reads the same
    # .dist-info metadata pip reads and works under conda, venv, and uv alike.
    python - <<'PY' 2>/dev/null > "$cache" || true
from importlib.metadata import distributions
for d in distributions():
    if d.name:
        print(d.name + "==" + d.version)
PY
  fi
  if [ ! -r "$cache" ]; then
    row badge_warn "$name" "Python $pyv · could not read installed packages"
    add_finding "warn" "$name" "pip install -r requirements.txt" "py_medi_watch_drift"
    return 2
  fi

  # Walk EVERY pinned dependency in requirements.txt (not a hand-picked subset)
  # and confirm pip freeze reports it installed, so the audit reflects the real
  # dependency surface. Commented lines (e.g. ray, apache-airflow) are skipped.
  local total=0 missing=0 first_missing=""
  while IFS= read -r line; do
    line="${line%%#*}"                       # drop inline + full-line comments
    local pkg="$line"
    pkg="${pkg%%=*}"; pkg="${pkg%%>*}"; pkg="${pkg%%<*}"
    pkg="${pkg%%\[*}"                        # drop [extras], e.g. python-jose[cryptography]
    pkg="${pkg%% *}"; pkg="${pkg//$'\t'/}"   # drop trailing spaces/tabs
    [ -z "$pkg" ] && continue
    total=$((total+1))
    # pip freeze normalizes hyphen/underscore inconsistently, so match either, and
    # accept '==', editable '@ url', or a bare name line.
    local pat="${pkg//[-_]/[-_]}"
    if ! grep -qiE "^${pat}(==| @ |$)" "$cache"; then
      missing=$((missing+1))
      [ -z "$first_missing" ] && first_missing="$pkg"
    fi
  done < "$req"

  if [ "$missing" -gt 0 ]; then
    local more=""
    [ "$missing" -gt 1 ] && more=" (+$((missing-1)) more)"
    row badge_warn "$name" "Python $pyv · $missing/$total missing: $first_missing$more"
    add_finding "warn" "$name" "pip install -r requirements.txt" "py_medi_watch_drift"
    return 2
  fi
  row badge_ok "$name" "Python $pyv · $total/$total installed"
  return 0
}

# Twelve runtime services, one row per image. Each entry is
#   "<service-label>|<image:tag>|<compose-service-for-ps>|<access-url>"
# where the compose-service is the canonical container we check for the
# "running" state (airflow has many services on one image, the api-server
# is the user-facing one, same for ray + ray-head).
#
# The core MLOps stack. Every service boots with plain `docker compose up -d`.
declare -a SERVICES_ORCH=(
  "postgres|postgres:18-alpine|postgres|tcp://localhost:5432"
  "airflow|mlops-airflow:latest|airflow-api-server|http://localhost:8080"
  "mlflow|mlops-mlflow:latest|mlflow|http://localhost:5000"
  "ray|mlops-ray:latest|ray-head|http://localhost:8265"
  "prometheus|prom/prometheus:v3.11.2|prometheus|http://localhost:9090"
  "grafana|grafana/grafana:13.0.1|grafana|http://localhost:3003"
  "inference-api|mlops-inference-api:latest|inference-api|http://localhost:8002"
  "driftly-api|mlops-driftly-api:latest|driftly-api|http://localhost:8003"
  "driftly-web|mlops-driftly-web:latest|driftly-web|http://localhost:3004"
)

# probe_services: one row per service, answering three questions at once:
#   image built?  container running?  what URL?
# Tally rules: built+running → OK, built+stopped → STOP (informational, no
# tally penalty), not built → MISS. Missing images add an install finding,
# stopped containers do not: STOP is a state, not a failure.
#
probe_services_subheading() {
  printf '\n  %s── %s ──%s\n' "$C_DIM" "$1" "$C_RESET"
}

probe_services_row() {
  # Probe one "<svc>|<image>|<cservice>|<url>" entry against the cached
  # running-service list passed in as $2. Hoisted out of the loop so every
  # service shares one per-row code path.
  local entry="$1" running_list="$2"
  local svc image cservice url
  IFS='|' read -r svc image cservice url <<< "$entry"

  local built=0 running=0
  docker image inspect "$image" >/dev/null 2>&1 && built=1
  if [ "$built" -eq 1 ] && printf '%s\n' "$running_list" | grep -qx "$cservice"; then
    running=1
  fi

  if [ "$running" -eq 1 ]; then
    row badge_ok   "$svc" "built · running   $url"
  elif [ "$built" -eq 1 ]; then
    row badge_stop "$svc" "built · stopped   $url"
  else
    row badge_miss "$svc" "not built"
    add_finding "miss" "$svc" "docker compose pull && docker compose build" "docker_images"
  fi
}

probe_services() {
  if ! command -v docker >/dev/null 2>&1; then
    row badge_miss "services" "docker not installed (probe skipped)"
    add_finding "miss" "services" "install docker first" "docker_images"
    return 1
  fi

  # Reachability check before the per-service loop. If `docker version` fails
  # while `docker` is on PATH, the daemon is down OR $USER lacks docker-group
  # membership in THIS shell. The latter is the common case right after
  # `doctor` (or `init`'s autofix) ran `inst_host_docker`: the new group
  # only takes effect in a fresh shell. We surface ONE actionable row
  # instead of 12 confusing "not built" MISSes.
  if ! docker version >/dev/null 2>&1; then
    if [ "${DOCKER_NEEDS_RELOGIN:-0}" -eq 1 ]; then
      row badge_miss "docker access" "$USER not in 'docker' group in THIS shell"
      note "open a new shell (or run: newgrp docker), then re-run audit/doctor"
      add_finding "miss" "docker access" "open a new shell to pick up the docker group" "docker_relogin"
    else
      row badge_miss "docker access" "daemon unreachable (down or permission denied)"
      note "try: docker version — if that fails, start dockerd or check group membership"
      add_finding "miss" "docker access" "docker daemon unreachable" "docker_unreachable"
    fi
    return 1
  fi

  # One shell-out for all running services, faster than 12 inspects.
  local running_list=""
  if [ -f "$INFRA/docker-compose.yml" ]; then
    running_list=$( ( cd "$INFRA" && docker compose ps --status running --services 2>/dev/null ) || true )
  fi

  local entry
  for entry in "${SERVICES_ORCH[@]}"; do
    probe_services_row "$entry" "$running_list"
  done
  return 0
}

# Required for default `init` (everything in the default compose profile).
# Missing/empty/placeholder values here → MISS.
declare -a ENV_KEYS_CORE=(
  POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB
  AIRFLOW_ADMIN_USER AIRFLOW_ADMIN_PASSWORD AIRFLOW_ADMIN_EMAIL
  AIRFLOW_SECRET_KEY AIRFLOW_FERNET_KEY AIRFLOW_UID
  HOST_IP
)

# env_key_state <file> <key>
# echoes one of: "ok" | "missing" | "empty" | "placeholder" | "tooshort"
# Used by probe_env_file to bucket each key into a state.
env_key_state() {
  local file="$1" key="$2"
  local line val
  line=$(grep -E "^${key}=" "$file" 2>/dev/null | head -1)
  if [ -z "$line" ]; then echo "missing"; return; fi
  val="${line#*=}"
  if [ -z "$val" ]; then echo "empty"; return; fi
  case "$val" in
    replace-me|hf_replace_me|change-me|changeme|placeholder|REPLACE_ME)
      echo "placeholder"; return ;;
  esac
  # specific forbidden values for individual keys
  case "$key" in
    AIRFLOW_ADMIN_PASSWORD)
      [ "$val" = "admin" ] && { echo "placeholder"; return; } ;;
  esac
  echo "ok"
}

# Walks an array of key names, prints one row per category, lists bad keys
# inline. Returns 0 if all OK, 1 if any bad.
_check_env_bucket() {
  local file="$1" label="$2" badge_bad="$3" category="$4"; shift 4
  local keys=("$@")
  local bad=() key state
  for key in "${keys[@]}"; do
    state=$(env_key_state "$file" "$key")
    [ "$state" = "ok" ] || bad+=("$key=$state")
  done
  if [ "${#bad[@]}" -eq 0 ]; then
    row badge_ok "$label" "${#keys[@]} keys present, all valid"
    return 0
  fi
  row "$badge_bad" "$label" "${#bad[@]}/${#keys[@]} bad: ${bad[*]}"
  local state_word="miss"
  [ "$badge_bad" = "badge_warn" ] && state_word="warn"
  add_finding "$state_word" "$label" "edit infra/.env (fill empty/placeholder keys)" "$category"
  return 1
}

probe_env_file() {
  local f="$INFRA/.env"
  if [ ! -f "$f" ]; then
    row badge_miss "infra/.env" "file not found"
    add_finding "miss" "infra/.env" "generate via 'mediwatch.sh init'" "config_env"
    return 1
  fi
  _check_env_bucket "$f" "config (core)"    badge_miss config_env_core    "${ENV_KEYS_CORE[@]}"
  return 0
}

# surface known doc/.env-example drift
probe_doc_drift() {
  local name="docs drift"
  local example="$INFRA/.env.example"
  local install_md="$REPO_ROOT/INSTALL.md"
  local issues=0 detail=""
  if [ -r "$install_md" ] && [ -r "$example" ]; then
    if grep -q 'AIRFLOW_JWT_SECRET' "$install_md" 2>/dev/null && \
       ! grep -q 'AIRFLOW_JWT_SECRET' "$example" 2>/dev/null; then
      issues=$((issues+1)); detail="${detail}INSTALL.md says AIRFLOW_JWT_SECRET; example uses AIRFLOW_SECRET_KEY. "
    fi
  fi
  if [ "$issues" -eq 0 ]; then
    return 0  # silent when no drift
  fi
  row badge_warn "$name" "$issues mismatch(es) between INSTALL.md and .env.example"
  note "$detail"
  add_finding "warn" "$name" "(informational; the wizard handles both correctly)" "docs_drift"
  return 2
}

# ============================================================================
# run_audit: run every probe, print summary, set exit code
# ============================================================================

run_audit() {
  FINDINGS=()
  OK_COUNT=0; WARN_COUNT=0; MISS_COUNT=0; STOP_COUNT=0; SKIP_COUNT=0

  hr_section "$C_CYAN" "HOST PREREQS"
  probe_docker          || true
  probe_compose         || true
  probe_nvidia_driver   || true
  probe_cuda            || true
  probe_nvidia_ctk      || true
  probe_python          || true
  probe_git             || true
  probe_bash            || true

  hr_section "$C_YELLOW" "PYTHON LIBS"
  probe_python_libs     || true

  hr_section "$C_MAGENTA" "SERVICES"
  probe_services        || true

  hr_section "$C_GREEN" "CONFIG"
  probe_env_file        || true
  probe_doc_drift       || true

  echo
  gradient_rule 63
  printf '  %s%d OK%s  ·  %s%d STOP%s  ·  %s%d SKIP%s  ·  %s%d WARN%s  ·  %s%d MISS%s\n' \
    "$C_GREEN" "$OK_COUNT"   "$C_RESET" \
    "$C_DIM"   "$STOP_COUNT" "$C_RESET" \
    "$C_DIM"   "$SKIP_COUNT" "$C_RESET" \
    "$C_AMBER" "$WARN_COUNT" "$C_RESET" \
    "$C_RED"   "$MISS_COUNT" "$C_RESET"
  gradient_rule 63
  echo

  # STOP is informational (container not started). Only WARN / MISS gate exit.
  if [ "$MISS_COUNT" -gt 0 ] || [ "$WARN_COUNT" -gt 0 ]; then
    return 1
  fi
  return 0
}

# ============================================================================
# installer registry + offer_installers: consent-driven autofix for the
# findings populated by run_audit.
#
# Every probe tags its add_finding call with a 'category' (host_docker,
# host_compose, py_medi_watch_drift, config_env, ...). installer_for maps a
# category to one of three handlers:
#
#   "yes"    → call <recipe-fn> after user consent. The sudo flag says whether
#              to prime sudo before running.
#   "manual" → print the finding's existing hint, user must act. Used for
#              fixes that need reboots, secret entry, or distro-specific repo
#              setup that we will not silently do for them.
#   ""       → no handler. Category is either handled elsewhere (docker_images
#              is built by `init` itself) or purely informational (docs_drift).
#
# offer_installers reads $FINDINGS, presents the actionable ones as a numbered
# pick-list, runs the selected recipes, then prints a summary. The caller is
# responsible for re-running run_audit afterward to refresh counts.
#
# Env contract:
#   INSTALL_DISABLED=1     → skip the whole phase (e.g. `init --no-fix`)
#   INSTALL_ASSUME_YES=1   → select all installable fixes without prompting
# ============================================================================

installer_for() {
  # On non-Linux hosts we never auto-install system packages (policy: degrade
  # to manual hints). The package categories become "manual" so the finding's
  # per-OS pkg_hint is shown instead of an apt recipe being offered.
  if [ "$OS_KIND" != "linux" ]; then
    case "$1" in
      host_docker|host_compose|host_git)
        echo "manual"$'\t'""$'\t'"0"; return ;;
    esac
  fi
  case "$1" in
    host_docker)         echo "yes"$'\t'"inst_host_docker"$'\t'"1" ;;
    host_compose)        echo "yes"$'\t'"inst_host_compose"$'\t'"1" ;;
    host_git)            echo "yes"$'\t'"inst_host_git"$'\t'"1" ;;
    host_nvdrv)          echo "manual"$'\t'""$'\t'"1" ;;
    host_cuda)           echo "manual"$'\t'""$'\t'"1" ;;
    host_nvctk)          echo "manual"$'\t'""$'\t'"1" ;;
    host_python)         echo "manual"$'\t'""$'\t'"0" ;;
    py_medi_watch)       echo "manual"$'\t'""$'\t'"0" ;;
    py_medi_watch_drift) echo "yes"$'\t'"inst_py_drift"$'\t'"0" ;;
    config_env)          echo "yes"$'\t'"inst_config_env"$'\t'"0" ;;
    config_env_core)     echo "manual"$'\t'""$'\t'"0" ;;
    *)                   echo "" ;;
  esac
}

# Run `apt-get update` at most once per offer_installers invocation. Recipes
# that use apt all call this first, the second and later calls are no-ops. On a
# non-Debian-derived host (no apt-get on PATH), bail with a clear message rather
# than letting `sudo apt-get update` print an opaque error.
APT_UPDATED=0
_apt_update_once() {
  command -v apt-get >/dev/null 2>&1 \
    || die "apt-get not available — autofix recipes only support Debian/Ubuntu hosts. Install the package via your distro's package manager, then re-run."
  if [ "$APT_UPDATED" -eq 0 ]; then
    sudo apt-get update -y
    APT_UPDATED=1
  fi
}

inst_host_docker() {
  hr "installing docker engine"
  info "running get.docker.com bootstrap (sudo)"
  curl -fsSL https://get.docker.com | sudo sh
  if id -nG "$USER" 2>/dev/null | grep -qw docker; then
    info "$USER is already in the docker group"
  else
    info "adding $USER to the docker group"
    sudo usermod -aG docker "$USER"
    # Surface the relogin requirement to probe_services so a subsequent
    # re-audit emits ONE actionable row instead of N stale "not built"
    # MISSes from compose calls that lack docker-group access yet.
    DOCKER_NEEDS_RELOGIN=1
    warn "log out + back in (or run 'newgrp docker') for the group change to take effect"
  fi
}

inst_host_compose() {
  hr "installing docker compose plugin"
  _apt_update_once
  sudo apt-get install -y docker-compose-plugin
}

inst_host_git() {
  hr "installing git"
  _apt_update_once
  sudo apt-get install -y git
}

inst_py_drift() {
  hr "syncing python deps"
  # Refuse to install into a bare system Python. Project convention (see
  # CLAUDE.md) is a uv-managed venv. Installing into system Python either
  # PEP-668-errors on modern Ubuntu or silently corrupts the host env.
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    warn "no venv active — refusing to install into system Python"
    warn "create one with 'uv venv && source .venv/bin/activate' then re-run doctor"
    return 1
  fi
  local env_label="${VIRTUAL_ENV##*/}"
  # uv installs into $VIRTUAL_ENV by default, so this targets the active env.
  # uv venvs ship without pip, so uv is the install path.
  if ! command -v uv >/dev/null 2>&1; then
    warn "uv not on PATH — install uv (curl -LsSf https://astral.sh/uv/install.sh | sh) then re-run"
    return 1
  fi
  info "uv pip install -r $ROOT/requirements.txt (target env: $env_label)"
  uv pip install -r "$ROOT/requirements.txt"
}

inst_config_env() {
  hr "seeding infra/.env"
  local src="$INFRA/.env.example"
  local dst="$INFRA/.env"
  if [ ! -f "$src" ]; then
    warn "no $src to copy from"
    return 1
  fi
  if [ -f "$dst" ]; then
    info "$dst already exists; leaving it alone"
    return 0
  fi
  cp "$src" "$dst"
  info "wrote $dst from .env.example"
  warn "open $dst and fill the placeholder secrets before 'init'"
}

# offer_installers: present autofix choices, run selected recipes.
offer_installers() {
  if [ "${INSTALL_DISABLED:-0}" -eq 1 ]; then
    return 0
  fi

  local installable=()
  local manual=()
  local f state name detail cat meta kind recipe sudo_flag

  for f in "${FINDINGS[@]:-}"; do
    [ -z "$f" ] && continue
    IFS=$'\t' read -r state name detail cat <<<"$f"
    case "$state" in miss|warn) ;; *) continue ;; esac
    meta=$(installer_for "$cat")
    [ -z "$meta" ] && continue
    IFS=$'\t' read -r kind recipe sudo_flag <<<"$meta"
    case "$kind" in
      yes)    installable+=("$name|$recipe|$sudo_flag|$detail") ;;
      manual) manual+=("$name|$detail") ;;
    esac
  done

  if [ "${#installable[@]}" -eq 0 ] && [ "${#manual[@]}" -eq 0 ]; then
    return 0
  fi

  hr_section "$C_CYAN" "AUTOFIX"

  if [ "${#installable[@]}" -gt 0 ]; then
    printf '  %sCan be installed automatically:%s\n\n' "$C_BOLD" "$C_RESET"
    local i p_name p_recipe p_sudo p_detail
    for i in "${!installable[@]}"; do
      IFS='|' read -r p_name p_recipe p_sudo p_detail <<<"${installable[$i]}"
      printf '    %s[%d]%s %-26s %s%s%s\n' \
        "$C_CYAN" "$((i+1))" "$C_RESET" "$p_name" "$C_DIM" "$p_detail" "$C_RESET"
    done
    echo

    local choice
    if [ "${INSTALL_ASSUME_YES:-0}" -eq 1 ]; then
      choice="all"
      info "(--yes assumed; selecting all)"
    elif [ "$IS_TTY" -eq 0 ]; then
      warn "non-TTY and --yes not set; skipping autofix"
      choice="none"
    else
      printf '  %sSelect:%s [a]ll · [n]one · comma list (e.g. 1,3): ' \
        "$C_BOLD" "$C_RESET"
      read -r choice
      [ -z "$choice" ] && choice="none"
    fi

    local selected=()
    case "$choice" in
      a|A|all|yes|y) selected=("${!installable[@]}") ;;
      n|N|none|no)   selected=() ;;
      *)
        local IFS_OLD="$IFS"
        IFS=','
        local part
        for part in $choice; do
          part="${part// /}"
          [ -z "$part" ] && continue
          if [[ "$part" =~ ^[0-9]+$ ]] && [ "$part" -ge 1 ] && [ "$part" -le "${#installable[@]}" ]; then
            selected+=("$((part-1))")
          else
            warn "ignoring invalid selection: $part"
          fi
        done
        IFS="$IFS_OLD"
        ;;
    esac

    if [ "${#selected[@]}" -gt 0 ]; then
      local needs_sudo=0 idx
      for idx in "${selected[@]}"; do
        IFS='|' read -r _ _ p_sudo _ <<<"${installable[$idx]}"
        [ "$p_sudo" = "1" ] && needs_sudo=1
      done
      SUDO_KEEPALIVE_PID=""
      if [ "$needs_sudo" -eq 1 ]; then
        info "priming sudo (one prompt for password)"
        sudo -v || die "sudo failed; aborting autofix"
        # Background keepalive so long recipes (docker pull, large apt
        # bundles) don't re-prompt mid-install once the sudo timestamp
        # expires. `-n` keeps the refresh non-interactive, and the loop
        # self-terminates the moment sudo can't refresh.
        ( while true; do sudo -n -v 2>/dev/null || exit; sleep 50; done ) &
        SUDO_KEEPALIVE_PID=$!
        # Survive Ctrl-C / die mid-loop without leaving the keepalive PID
        # behind. Re-entry is safe: the trap re-reads SUDO_KEEPALIVE_PID
        # every time, and we clear it after a clean shutdown below.
        trap '[ -n "${SUDO_KEEPALIVE_PID:-}" ] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null; SUDO_KEEPALIVE_PID=""' EXIT INT TERM
      fi

      local ok_n=0 fail_n=0 failed_names=()
      for idx in "${selected[@]}"; do
        IFS='|' read -r p_name p_recipe p_sudo p_detail <<<"${installable[$idx]}"
        printf '\n  %s→ installing %s%s%s\n' "$C_CYAN" "$C_BOLD" "$p_name" "$C_RESET"
        set +e
        "$p_recipe"
        local rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then
          ok_n=$((ok_n+1))
          printf '  %s✓ %s installed%s\n' "$C_GREEN" "$p_name" "$C_RESET"
        else
          fail_n=$((fail_n+1))
          failed_names+=("$p_name")
          printf '  %s✗ %s failed (exit %d)%s\n' "$C_RED" "$p_name" "$rc" "$C_RESET"
        fi
      done

      # Stop the sudo keepalive now that the recipe loop is done.
      if [ -n "${SUDO_KEEPALIVE_PID:-}" ]; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        SUDO_KEEPALIVE_PID=""
      fi

      echo
      printf '  %sautofix:%s %d installed · %d failed' \
        "$C_BOLD" "$C_RESET" "$ok_n" "$fail_n"
      if [ "$fail_n" -gt 0 ]; then
        printf ' (%s)' "${failed_names[*]}"
      fi
      echo
    fi
  fi

  if [ "${#manual[@]}" -gt 0 ]; then
    echo
    printf '  %sNeeds manual attention:%s\n\n' "$C_BOLD" "$C_RESET"
    local entry p_name p_detail
    for entry in "${manual[@]}"; do
      IFS='|' read -r p_name p_detail <<<"$entry"
      printf '    %s· %s%s%s — %s%s%s\n' \
        "$C_AMBER" "$C_BOLD" "$p_name" "$C_RESET" "$C_DIM" "$p_detail" "$C_RESET"
    done
  fi
  echo
}

# ============================================================================
# preflight_audit: run audit before any state-mutating command.
#
# Usage: preflight_audit <cmd-label> [<promoted-finding-category>...]
#
# Behavior:
#   1. Run audit.
#   2. If anything is MISS or WARN (and INSTALL_DISABLED != 1), invoke
#      offer_installers and re-audit so the gate evaluates the post-fix state.
#   3. Apply the original gate:
#        MISS_COUNT > 0                                  → die
#        any FINDING category matches promoted args      → die
#        WARN_COUNT > 0 and no promotion match           → prompt
#        else                                            → "preflight clean"
#
# Env contract (set by callers):
#   PREFLIGHT_ASSUME_YES=1            → skip the WARN proceed prompt
#   INSTALL_ASSUME_YES=1              → autofix selects all without prompting
#   INSTALL_DISABLED=1                → skip the autofix phase entirely
#   PREFLIGHT_IGNORE_MISS_CATEGORIES  → space-separated categories whose MISS
#                                        the caller will resolve itself
# ============================================================================

preflight_audit() {
  local cmd_label="$1"; shift
  local promoted=("$@")

  hr_section "$C_CYAN" "PREFLIGHT AUDIT (before $cmd_label)"
  run_audit || true

  # Try to repair before gating. If anything got installed,
  # re-audit so the rest of this function sees the post-fix state.
  if [ "${INSTALL_DISABLED:-0}" -ne 1 ] \
     && { [ "$MISS_COUNT" -gt 0 ] || [ "$WARN_COUNT" -gt 0 ]; }; then
    offer_installers
    hr_section "$C_CYAN" "RE-AUDIT (after autofix)"
    run_audit || true
  fi

  # any FINDINGS in a promoted category turn into a hard fail for this cmd
  local promoted_hits=()
  if [ "${#promoted[@]}" -gt 0 ]; then
    local f cat fcat
    for f in "${FINDINGS[@]:-}"; do
      [ -z "$f" ] && continue
      fcat="${f##*$'\t'}"
      for cat in "${promoted[@]}"; do
        [ "$fcat" = "$cat" ] && promoted_hits+=("$f")
      done
    done
  fi

  # MISS findings in PREFLIGHT_IGNORE_MISS_CATEGORIES (space-separated) are
  # demoted: the calling command guarantees it will resolve them itself
  # (e.g. `init` runs `docker compose up -d`, which builds any missing
  # mlops-* images, so a "not built" MISS is not a real blocker).
  local effective_miss=0
  local addressed_findings=()
  local ignore_set=" ${PREFLIGHT_IGNORE_MISS_CATEGORIES:-} "
  if [ "$MISS_COUNT" -gt 0 ]; then
    local f fstate fname fdet fcat
    for f in "${FINDINGS[@]:-}"; do
      [ -z "$f" ] && continue
      IFS=$'\t' read -r fstate fname fdet fcat <<<"$f"
      [ "$fstate" = "miss" ] || continue
      if [[ "$ignore_set" == *" $fcat "* ]]; then
        addressed_findings+=("$fname")
      else
        effective_miss=$((effective_miss+1))
      fi
    done
  fi

  if [ "$effective_miss" -gt 0 ]; then
    printf '\n  %s✗ preflight failed%s — %d MISS finding(s) must be fixed before %s.\n' \
      "$C_RED" "$C_RESET" "$effective_miss" "$cmd_label"
    printf '  Resolve the issues above (run %s./mediwatch.sh doctor%s for a full audit + autofix)\n' \
      "$C_CYAN" "$C_RESET"
    printf '  or edit %sinfra/.env%s manually, then re-run %s./mediwatch.sh %s%s.\n\n' \
      "$C_CYAN" "$C_RESET" "$C_CYAN" "$cmd_label" "$C_RESET"
    exit 1
  fi

  if [ "${#addressed_findings[@]}" -gt 0 ]; then
    printf '\n  %s↪ %d MISS finding(s) will be addressed by %s%s%s:\n' \
      "$C_CYAN" "${#addressed_findings[@]}" "$C_BOLD" "$cmd_label" "$C_RESET"
    local item
    for item in "${addressed_findings[@]}"; do
      printf '    %s· %s%s\n' "$C_DIM" "$item" "$C_RESET"
    done
  fi

  if [ "${#promoted_hits[@]}" -gt 0 ]; then
    printf '\n  %s✗ preflight failed%s — %d critical issue(s) for %s%s%s:\n' \
      "$C_RED" "$C_RESET" "${#promoted_hits[@]}" "$C_BOLD" "$cmd_label" "$C_RESET"
    local hit fname fdet
    for hit in "${promoted_hits[@]}"; do
      IFS=$'\t' read -r _ fname fdet _ <<< "$hit"
      printf '    %s· %s%s%s — %s\n' "$C_RED" "$C_BOLD" "$fname" "$C_RESET" "$fdet"
    done
    printf '  These are normally warnings, but %s%s%s requires them to be set.\n\n' \
      "$C_BOLD" "$cmd_label" "$C_RESET"
    exit 1
  fi

  if [ "$WARN_COUNT" -gt 0 ]; then
    printf '\n  %s⚠ preflight passed with %d warning(s).%s\n' \
      "$C_AMBER" "$WARN_COUNT" "$C_RESET"
    if [ "${PREFLIGHT_ASSUME_YES:-0}" -eq 1 ]; then
      printf '  %s(--yes assumed; proceeding despite warnings)%s\n\n' "$C_DIM" "$C_RESET"
    elif [ "$IS_TTY" -eq 0 ]; then
      die "non-TTY and --yes not provided; refusing to proceed with warnings"
    else
      local reply
      read -r -p "  Proceed anyway? Type 'yes' to continue: " reply
      [ "$reply" = "yes" ] || die "aborted"
    fi
  else
    printf '\n  %s✓ preflight clean%s — proceeding with %s\n\n' \
      "$C_GREEN" "$C_RESET" "$cmd_label"
  fi
}

# ============================================================================
# verbose-stage helper (used by init/reset)
# ============================================================================

hr() { printf '\n%s== %s ==%s\n' "$C_CYAN" "$*" "$C_RESET"; }

# Shared pre-destruction explainer. Both `init --reset` and `shutdown --reset`
# print this exact list before prompting, so the user always sees the same
# destructive plan no matter which command they ran.
_print_reset_warning() {
  printf '  %s%s--reset%s will wipe ALL state for this project:%s\n' "$C_AMBER" "$C_BOLD" "$C_RESET$C_AMBER" "$C_RESET"
  echo "    - docker compose down -v --remove-orphans"
  echo "    - remove EVERY volume labeled for compose project 'mlops' (named +"
  echo "      anonymous), erasing:"
  echo "        · postgres-db      → Airflow metadata + MLflow runs/experiments"
  echo "        · mlflow-artifacts → MLflow models & artifacts"
  echo "        · ray-tmp          → Ray / RayTune session + object store"
  echo "        · prom-data, grafana-data → Prometheus TSDB + Grafana state"
  echo "    - remove every image declared in infra/docker-compose.yml — both"
  echo "      locally-built (mlops-*:latest) and pulled (postgres, grafana,"
  echo "      prom/prometheus, nginx). Next 'init' re-pulls and rebuilds."
  echo "    - rm -rf $ROOT/data  (then re-create with correct ownership)"
  echo "    - clear $INFRA/airflow/logs/  (all task/run logs; keeps .gitkeep)"
  echo "    - rm -rf any stray $ROOT/{mlruns,ray_results,postgres-data}"
  echo
}

# ============================================================================
# do_reset_teardown: destructive wipe of this project's compose state.
#
# Removes only THIS project's resources: every volume labeled for compose
# project 'mlops' (named + anonymous), every image declared in the compose
# file, data/, airflow logs, and stray host state dirs. Does NOT run
# `docker system prune`, which nukes resources from other projects on the same
# host. Caller is responsible for the consent prompt. This function only executes.
# ============================================================================

do_reset_teardown() {
  # --rmi local clears any compose-managed images that don't have a custom
  # tag (anonymous build:-only services, intermediate layers). Our explicit
  # `image: mlops-<svc>:latest` tags ARE custom, so --rmi local won't catch
  # them. The dedicated rmi pass below handles those.
  hr "docker compose down --rmi local -v --remove-orphans"
  ( cd "$INFRA" && docker compose down --rmi local -v --remove-orphans ) || \
    warn "compose down returned non-zero; continuing teardown anyway"

  hr "remove ALL volumes for compose project 'mlops' (idempotent)"
  # Label-scoped removal is strictly more complete than matching mlops_* by
  # name: compose stamps every volume it creates — named AND anonymous (build
  # scratch, unnamed mounts with hash names) — with this label. This is what
  # actually erases the persisted state per service:
  #   postgres-db      → Airflow metadata DB + MLflow experiment/run metadata
  #   mlflow-artifacts → MLflow models/artifacts
  #   ray-tmp          → Ray session + RayTune object store under /tmp/ray
  #   prom-data        → Prometheus TSDB    grafana-data → Grafana state
  local proj_vols
  mapfile -t proj_vols < <(docker volume ls -q \
    --filter label=com.docker.compose.project=mlops 2>/dev/null)
  for vol in "${proj_vols[@]}"; do
    [ -n "$vol" ] || continue
    docker volume rm "$vol" >/dev/null 2>&1 && info "removed $vol" \
      || warn "failed to remove volume $vol (still in use?)"
  done
  # Fallback by explicit name in case the label filter resolved nothing (e.g.
  # volumes orphaned by a partial prior run that lost their project label).
  for vol in mlops_postgres-db mlops_mlflow-artifacts \
             mlops_ray-tmp mlops_prom-data mlops_grafana-data; do
    if docker volume inspect "$vol" >/dev/null 2>&1; then
      docker volume rm "$vol" >/dev/null 2>&1 && info "removed $vol (by name)"
    fi
  done

  hr "remove every image declared in this project's compose file"
  # `docker compose config --images` resolves variable substitution and emits
  # the canonical image list: both locally-built (mlops-*:latest) and pulled
  # (postgres, grafana, prom/prometheus, nginx).
  #
  # This is scoped to images THIS project declares. We still don't run
  # `docker system prune -a`. An image in use by another project's container
  # fails rmi and we warn (no force-cascade across projects).
  mapfile -t imgs < <( cd "$INFRA" && docker compose config --images 2>/dev/null | sort -u )
  if [ "${#imgs[@]}" -eq 0 ]; then
    # Empty list almost always means `docker compose config` failed silently
    # (most often: malformed .env or missing variable). The teardown can
    # still finish other phases, but the user should know images may linger.
    warn "no images resolved from compose config — likely .env parse failure; pulled images may remain on the host"
  else
    info "found ${#imgs[@]} project image(s):"
    for img in "${imgs[@]}"; do
      printf '    %s\n' "$img"
    done
    for img in "${imgs[@]}"; do
      # Skip images that aren't pulled/built: nothing to remove.
      docker image inspect "$img" >/dev/null 2>&1 || { info "not present, skipping $img"; continue; }
      # Loop one-by-one with `|| warn` so a single in-use image (e.g. still
      # referenced by a stray container from another project) doesn't abort
      # the whole teardown via `set -e`.
      if docker rmi -f "$img" >/dev/null 2>&1; then
        info "removed $img"
      else
        warn "failed to remove $img (still in use? try: docker ps -a | grep $img)"
      fi
    done
  fi

  hr "wipe data/ directory"
  if [ -d "$ROOT/data" ]; then
    info "rm -rf $ROOT/data"
    if ! rm -rf "$ROOT/data" 2>/dev/null; then
      warn "data/ is owned by another user (likely root from a prior docker run); using sudo"
      sudo rm -rf "$ROOT/data"
    fi
  fi
  info "mkdir -p $ROOT/data  (owned by $(id -un):$(id -gn), mode 775)"
  mkdir -p "$ROOT/data"
  chmod 775 "$ROOT/data"

  hr "clear airflow logs"
  local log_dir="$INFRA/airflow/logs"
  if [ -d "$log_dir" ]; then
    if ! find "$log_dir" -mindepth 1 -delete 2>/dev/null; then
      warn "need sudo to remove airflow-owned log files"
      sudo find "$log_dir" -mindepth 1 -delete
    fi
  fi
  # Restore the tracked sentinel so the dir stays in git and `init` can chown it.
  mkdir -p "$log_dir" && touch "$log_dir/.gitkeep"

  hr "remove stray host-side state dirs (idempotent)"
  # Belt-and-suspenders for state that older runs (or the compose comment at the
  # top of docker-compose.yml) may have left on the host even though current
  # services keep everything in named volumes. Only acts on dirs that exist.
  for d in mlruns ray_results postgres-data; do
    local stray="$ROOT/$d"
    [ -d "$stray" ] || continue
    info "rm -rf $stray"
    if ! rm -rf "$stray" 2>/dev/null; then
      warn "$d is owned by another user (likely root from a prior docker run); using sudo"
      sudo rm -rf "$stray"
    fi
  done
}

# ============================================================================
# cmd_init: bring docker compose services up. Does NOT run notebooks or
# pipelines. Optionally tears down state first (--reset).
# ============================================================================

cmd_init_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh init [--reset] [--yes|-y] [--no-fix] [--k8s] [--help]

Bring docker compose services up. Does NOT run notebooks or pipelines.

A full audit ALWAYS runs first (preflight). If MISS or WARN findings are
present, init offers to install the fixable ones with your consent before
the gate runs. Behavior of the preflight gate (after autofix):
  - MISS in "service not built"  → demoted (init builds it via `compose up`)
  - any other MISS finding       → init aborts
  - any WARN finding             → prompt to continue (--yes auto-accepts)
  - clean                        → proceed silently

Options:
  --reset      Full teardown before starting — wipes ALL project state:
                 - docker compose down -v --remove-orphans
                 - remove EVERY volume labeled for compose project 'mlops'
                   (named + anonymous): Airflow metadata + MLflow runs
                   (postgres-db), MLflow artifacts (mlflow-artifacts),
                   Ray/RayTune (ray-tmp), Prometheus + Grafana state.
                 - rm -rf data/   (then re-create with correct ownership)
                 - clear infra/airflow/logs/  (all task/run logs; keeps .gitkeep)
                 - rm -rf any stray ./{mlruns,ray_results,postgres-data}
                 - remove every image declared in infra/docker-compose.yml:
                   built (mlops-*) AND pulled (postgres, grafana, ...).
                   Next 'up' re-pulls and rebuilds.
  --yes, -y    Skip ALL prompts: preflight WARN confirmation, the autofix
               selection (selects all installable fixes), and --reset.
  --no-fix     Skip the autofix prompt entirely; gate runs on the raw audit.
  --k8s        After the compose stack is up, also deploy the inference API to a
               local minikube cluster (bootstraps minikube if needed). This is
               the production topology — inference served from k8s, redeployed by
               the retrain DAG on a new @champion. Non-blocking: pods reach Ready
               only after a train/retrain run registers a champion. Requires
               minikube + docker. Default init stays compose-only.

Examples:
  ./mediwatch.sh init                    # audit, offer fixes, then up
  ./mediwatch.sh init --no-fix           # audit, gate immediately (no autofix)
  ./mediwatch.sh init --reset            # preflight, nuke state, then up (prompts)
  ./mediwatch.sh init --reset --yes      # CI-style: wipe + no prompts
  ./mediwatch.sh init --k8s              # up, then deploy inference to minikube
EOF
}

# _ensure_airflow_log_perms: ensure infra/airflow/logs is writable by the
# airflow container's UID.
#
# Why this exists: the airflow-dag-processor lazily creates per-DAG-file log
# subdirs (e.g. /opt/airflow/logs/dag_processor) the first time it parses
# each file. If the host-mounted logs/ is owned by root (the common case
# when Docker auto-creates the bind-mount target before AIRFLOW_UID is
# resolved), that mkdir hits EACCES, the dag-processor crashes, and the UI
# silently shows zero DAGs. Fixing it once isn't enough: a `--reset` cycle
# or any process that re-touches the dir as root re-introduces the drift.
# So `init` runs this check unconditionally as part of the bring-up.
#
# Resolution order for the target UID: $AIRFLOW_UID in infra/.env  →  1000.
# (1000 matches the project's .env.example. 50000 is Airflow's upstream
# default, but we don't fall back to it here because the project pins UID 1000.)
_ensure_airflow_log_perms() {
  local logs_dir="$INFRA/airflow/logs"
  mkdir -p "$logs_dir"

  # Docker Desktop (macOS/Windows) remaps a bind mount to the container UID
  # ONLY when the host dir is owned by the host user. When the dir is missing
  # at first `up`, the Docker VM auto-creates it as root, and a root-owned bind
  # mount is NOT remapped: the container's uid 1000 then hits EACCES creating
  # logs/dag_processor and the dag-processor crash-loops (the symptom plugins/,
  # which ships a tracked .gitkeep and is host-user-owned, avoids). So on
  # Docker Desktop we don't chown to AIRFLOW_UID, we only guarantee the dir
  # exists owned by the host user. BSD `stat -f` is used (GNU `stat -c` below
  # is Linux-only).
  if [ "$OS_KIND" != "linux" ]; then
    local owner_uid
    owner_uid=$(stat -f '%u' "$logs_dir" 2>/dev/null || echo unknown)
    if [ "$owner_uid" = "0" ]; then
      info "airflow/logs is root-owned (Docker auto-created it) — recreating as $(id -un)"
      if [ -z "$(ls -A "$logs_dir" 2>/dev/null)" ] && rmdir "$logs_dir" 2>/dev/null; then
        mkdir -p "$logs_dir" && touch "$logs_dir/.gitkeep"
        info "recreated airflow/logs owned by $(id -un) — restart airflow containers to re-resolve the mount"
      else
        warn "airflow/logs is root-owned and not empty; cannot recreate without sudo"
        warn "  sudo chown -R $(id -u) $logs_dir"
      fi
    fi
    return 0
  fi

  local env_file="$INFRA/.env"
  local airflow_uid="1000"
  if [ -f "$env_file" ]; then
    local v
    v=$(grep -E '^AIRFLOW_UID=' "$env_file" 2>/dev/null | head -1 | cut -d= -f2-)
    [ -n "$v" ] && airflow_uid="$v"
  fi

  mkdir -p "$logs_dir"

  local current_uid
  current_uid=$(stat -c '%u' "$logs_dir" 2>/dev/null || echo unknown)
  if [ "$current_uid" = "$airflow_uid" ]; then
    return 0
  fi

  info "airflow/logs owned by UID $current_uid (expected $airflow_uid for AIRFLOW_UID)"
  info "fixing now to prevent airflow-dag-processor crash-loop"

  # Try without sudo first, which works if the user already owns the tree.
  if chown -R "$airflow_uid:$airflow_uid" "$logs_dir" 2>/dev/null; then
    info "chown succeeded (no sudo needed)"
    return 0
  fi

  # Fall back to sudo. Only proceed if a sudo session is already cached,
  # so we never block a non-TTY caller (CI) waiting for a password.
  if sudo -n true 2>/dev/null; then
    sudo chown -R "$airflow_uid:$airflow_uid" "$logs_dir"
    info "chown succeeded (via sudo)"
    return 0
  fi

  warn "need sudo to chown $logs_dir to UID $airflow_uid"
  warn "in another shell, run:"
  warn "  sudo chown -R $airflow_uid:$airflow_uid $logs_dir"
  warn "then re-run ./mediwatch.sh init"
  die "airflow-dag-processor would crash without writable logs/ (DAGs would not appear in UI)"
}

# _ensure_data_dir_perms: guarantee data/ exists and is host-user-writable
# BEFORE `docker compose up`. This is the permanent fix for the recurring
# "PermissionError: [Errno 13] Permission denied: '../data/cleaned.csv'" the
# host data-prep pipeline (./mediwatch.sh train) hits.
#
# Root cause: when a bind-mount SOURCE dir is missing at `up` time, the Docker
# daemon (root) creates it as root:root, independent of the container's `user:`
# setting, which only governs files written INSIDE the container. data/ is
# gitignored and ephemeral (wiped by --reset, or never committed), so any init
# after it goes missing lets dockerd recreate it root-owned, and the host
# pipeline (uid 1000) then can't write into it. Pre-creating the dir host-owned
# means dockerd finds an existing dir and leaves ownership alone, and the
# airflow container (which mounts it rw and runs as AIRFLOW_UID:0 = 1000:0)
# keeps every file it writes owned by the host user too, so it never re-roots.
# Run unconditionally in init, before bring-up. Mirrors _ensure_airflow_log_perms.
_ensure_data_dir_perms() {
  local data_dir="$ROOT/data"

  # Missing → create it as the host user (this process is the host user), so
  # dockerd never gets the chance to auto-create it as root. This line makes
  # the fix permanent: every init re-establishes the dir host-owned.
  if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir"
    chmod 775 "$data_dir"
    info "created $data_dir owned by $(id -un):$(id -gn) (mode 775)"
    return 0
  fi

  # Exists → probe writability directly rather than guessing from ownership
  # (an ACL or group-write could make a non-owned dir writable, and vice versa).
  if ( : > "$data_dir/.write_probe" ) 2>/dev/null; then
    rm -f "$data_dir/.write_probe" 2>/dev/null
    return 0
  fi

  info "$data_dir is not writable by $(id -un) — Docker likely auto-created it as root"
  info "reclaiming ownership (one-time; init keeps it host-owned from here on)"

  # Best case, NO sudo: a root-owned but EMPTY dir can be removed and recreated
  # host-owned, because deleting a directory entry needs write+execute on the
  # PARENT (which the host user owns), not ownership of the dir itself. This
  # keeps the steady state sudo-free even after a daemon auto-create.
  if [ -z "$(ls -A "$data_dir" 2>/dev/null)" ] && rmdir "$data_dir" 2>/dev/null; then
    mkdir -p "$data_dir"
    chmod 775 "$data_dir"
    info "recreated $data_dir owned by $(id -un):$(id -gn) (no sudo needed)"
    return 0
  fi

  # Non-empty: try chown without sudo (works if we own it but perms were tight).
  if chown -R "$(id -u):$(id -g)" "$data_dir" 2>/dev/null && chmod 775 "$data_dir" 2>/dev/null; then
    info "ownership reclaimed (no sudo needed)"
    return 0
  fi

  # Fall back to a CACHED sudo session only, never blocking a non-TTY caller (CI).
  if sudo -n true 2>/dev/null; then
    sudo chown -R "$(id -u):$(id -g)" "$data_dir" && chmod 775 "$data_dir"
    info "ownership reclaimed (via sudo)"
    return 0
  fi

  warn "$data_dir is root-owned and needs a ONE-TIME ownership reclaim."
  warn "in another shell, run:"
  warn "  sudo chown -R $(id -u):$(id -g) $data_dir"
  warn "then re-run ./mediwatch.sh init. After this once, init keeps it host-owned"
  warn "automatically and the PermissionError will not recur."
  die "host data-prep pipeline (and airflow's data writes) would fail without writable data/"
}

cmd_init() {
  local do_reset=0 assume_yes=0 no_fix=0 do_k8s=0
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_init_help; return 0 ;;
      --reset)   do_reset=1 ;;
      --yes|-y)  assume_yes=1 ;;
      --no-fix)  no_fix=1 ;;
      --k8s|--k8) do_k8s=1 ;;
      *) echo "init: unknown argument: $arg" >&2
         echo "run './mediwatch.sh init --help' for usage." >&2
         exit 2 ;;
    esac
  done

  command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
  [ -d "$INFRA" ] || die "infra/ directory not found at $INFRA"
  [ -f "$INFRA/docker-compose.yml" ] || die "infra/docker-compose.yml missing"

  print_banner "init"

  # Preflight audit. Always runs before any state change.
  # `docker_images` MISS findings ("service not built") are demoted: `init`
  # runs `docker compose up -d`, which builds them. Hard-failing here would
  # make `init` impossible to run on a fresh checkout.
  PREFLIGHT_ASSUME_YES=$assume_yes
  INSTALL_ASSUME_YES=$assume_yes
  INSTALL_DISABLED=$no_fix
  PREFLIGHT_IGNORE_MISS_CATEGORIES="docker_images"
  preflight_audit "init"

  if [ "$do_reset" -eq 1 ]; then
    _print_reset_warning
    if [ "$assume_yes" -ne 1 ]; then
      if [ "$IS_TTY" -eq 0 ]; then
        die "non-TTY and --yes not provided; refusing destructive reset"
      fi
      local reply
      read -r -p "  Proceed? Type 'yes' to continue: " reply
      [ "$reply" = "yes" ] || die "aborted"
    else
      printf '  %s(--yes assumed)%s\n' "$C_DIM" "$C_RESET"
    fi
    do_reset_teardown
  fi

  hr "ensuring airflow/logs is writable by AIRFLOW_UID"
  _ensure_airflow_log_perms

  hr "ensuring data/ exists and is host-writable (prevents pipeline PermissionError)"
  _ensure_data_dir_perms

  # Pick base-only vs base+GPU-overlay for this host. Sets COMPOSE_FILE, which
  # the `docker compose` calls below inherit. On GPU-less hosts (Mac, CPU Linux)
  # the overlay is omitted and the stack runs CPU-only.
  select_compose_files
  if _have_usable_gpu; then
    info "NVIDIA GPU detected — including docker-compose.gpu.yml overlay"
  else
    info "no usable NVIDIA GPU — running CPU-only (GPU overlay skipped)"
  fi

  # --k8s: minikube MUST be up BEFORE compose. The airflow worker joins
  # minikube's docker network (declared `external: true` in docker-compose.yml)
  # so its in-container kubectl can drive the k8s redeploy path. Compose
  # validates every declared network before creating ANY container, so if the
  # external `minikube` network is absent, `compose up` aborts and nothing
  # starts. `minikube start` creates that network, so it has to run first.
  if [ "$do_k8s" -eq 1 ]; then
    echo
    hr "init --k8s: bootstrap minikube first (compose worker joins its network)"
    _k8s_minikube_bootstrap
  fi

  hr "docker compose up -d"
  ( cd "$INFRA" && docker compose up -d )

  info "waiting 10s for services to settle..."
  sleep 10
  ( cd "$INFRA" && docker compose ps )

  # --k8s: now that the compose MLflow is up, deploy the inference API onto the
  # minikube cluster bootstrapped above (the production topology serves it from
  # k8s, not compose). Non-blocking on readiness: a fresh stack has no @champion
  # yet, so the pods go Ready only after a train/retrain run registers one.
  # Points at the compose MLflow via the host gateway (published on :5000).
  if [ "$do_k8s" -eq 1 ]; then
    echo
    hr "init --k8s: deploy inference API to minikube (non-blocking)"
    MLFLOW_TRACKING_URI="http://host.minikube.internal:5000" \
    ROLLOUT_WAIT="false" \
      _k8s_deploy_minikube_local "$K8S_NAMESPACE_DEFAULT"
  fi

  echo
  printf '  %s✓ stack is up%s\n' "$C_GREEN" "$C_RESET"
  printf '  MLflow:     %shttp://localhost:5000%s\n' "$C_CYAN" "$C_RESET"
  printf '  Airflow:    %shttp://localhost:8080%s\n' "$C_CYAN" "$C_RESET"
  printf '  Grafana:    %shttp://localhost:3003%s\n' "$C_CYAN" "$C_RESET"
  printf '  Prometheus: %shttp://localhost:9090%s\n' "$C_CYAN" "$C_RESET"
  printf '  Ray:        %shttp://localhost:8265%s\n' "$C_CYAN" "$C_RESET"
  printf '  Inference:  %shttp://localhost:8002%s\n' "$C_CYAN" "$C_RESET"
  printf '  Driftly:    %shttp://localhost:3004%s  %s(drift dashboard · API :8003)%s\n' \
    "$C_CYAN" "$C_RESET" "$C_DIM" "$C_RESET"
  if [ "$do_k8s" -eq 1 ]; then
    printf '  Inference (k8s): %skubectl port-forward svc/medi-watch-inference 8002:80 -n %s%s\n' \
      "$C_CYAN" "$K8S_NAMESPACE_DEFAULT" "$C_RESET"
    printf '            %s(pods reach Ready after a train/retrain registers @champion)%s\n' \
      "$C_DIM" "$C_RESET"
  fi
  printf '  run %s./mediwatch.sh doctor%s (or doctor --no-fix for read-only) to re-audit.\n\n' "$C_CYAN" "$C_RESET"
}

# ============================================================================
# cmd_shutdown: halt containers. With --reset, wipes everything (containers,
# volumes, ALL project images, data/, airflow logs). Without --reset, only
# halts. Does not run a preflight audit (stopping needs no config).
# ============================================================================

cmd_shutdown_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh shutdown [--reset] [--yes|-y] [--help]

Halt docker compose services.

Default (no --reset):
  docker compose stop  — halts running containers in place. Volumes, images,
  and data/ are preserved. Bring them back with './mediwatch.sh init'
  (a fresh 'up' that recreates the containers from the unchanged images).
  A running local minikube cluster is powered off too (resume: minikube start).

Options:
  --reset      Full teardown — DESTRUCTIVE. Wipes ALL state for this project.
               Same as 'init --reset' teardown, but without bringing the stack
               back up afterward:
                 - docker compose down -v --remove-orphans
                 - remove EVERY volume labeled for compose project 'mlops'
                   (named + anonymous), erasing all persisted data:
                     · Airflow metadata + MLflow runs/experiments (postgres-db)
                     · MLflow models & artifacts (mlflow-artifacts)
                     · Ray / RayTune session + object store (ray-tmp)
                     · Prometheus TSDB + Grafana state (prom-data, grafana-data)
                 - rm -rf data/   (then re-create with correct ownership)
                 - clear infra/airflow/logs/  (all task/run logs; keeps .gitkeep)
                 - rm -rf any stray ./{mlruns,ray_results,postgres-data}
                 - remove every image declared in infra/docker-compose.yml:
                   built (mlops-*) AND pulled (postgres, grafana, ...).
                 - delete the medi-watch namespace and the minikube cluster.
  --yes, -y    Skip the --reset confirmation prompt.

Examples:
  ./mediwatch.sh shutdown                # halt; resume with 'init'
  ./mediwatch.sh shutdown --reset        # wipe everything (prompts)
  ./mediwatch.sh shutdown --reset --yes  # wipe with no prompt
EOF
}

# _shutdown_k8s: tear down the local minikube k8s side that docker compose does
# not own. `docker compose stop` halts the Compose services but leaves the
# minikube cluster and the medi-watch namespace running, so without this step
# the inference pods keep serving and the cluster keeps consuming resources
# after a shutdown. mode=stop powers the cluster off and keeps it on disk for a
# fast resume, parallel to `docker compose stop`. mode=delete removes the
# workload namespace and the cluster itself, parallel to a full compose wipe.
# Both are best-effort: a host without minikube or kubectl is a valid setup and
# must not turn shutdown into a failure.
_shutdown_k8s() {
  local mode="$1"
  if ! command -v minikube >/dev/null 2>&1; then
    info "minikube not installed, no local k8s cluster to tear down"
    return 0
  fi

  local running=0
  minikube status 2>/dev/null | grep -q 'host: Running' && running=1

  if [ "$mode" = "delete" ]; then
    # Delete the workload namespace first when the API server is reachable, so
    # the cluster records a clean removal before the cluster itself is gone.
    if [ "$running" -eq 1 ] && command -v kubectl >/dev/null 2>&1 \
       && kubectl cluster-info >/dev/null 2>&1; then
      hr "kubectl delete namespace $K8S_NAMESPACE_DEFAULT"
      kubectl delete namespace "$K8S_NAMESPACE_DEFAULT" --ignore-not-found --wait=false 2>/dev/null \
        || warn "namespace delete returned non-zero, continuing to cluster delete"
    fi
    hr "minikube delete"
    minikube delete 2>&1 | tail -4 || warn "minikube delete returned non-zero"
    info "minikube cluster removed"
    return 0
  fi

  # mode=stop
  if [ "$running" -eq 1 ]; then
    hr "minikube stop"
    minikube stop 2>&1 | tail -3 || warn "minikube stop returned non-zero"
    info "minikube cluster powered off, resume with: minikube start"
  else
    info "minikube cluster not running, nothing to stop"
  fi
}

cmd_shutdown() {
  local do_reset=0 assume_yes=0
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_shutdown_help; return 0 ;;
      --reset)   do_reset=1 ;;
      --yes|-y)  assume_yes=1 ;;
      *) echo "shutdown: unknown argument: $arg" >&2
         echo "run './mediwatch.sh shutdown --help' for usage." >&2
         exit 2 ;;
    esac
  done

  command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
  [ -d "$INFRA" ] || die "infra/ directory not found at $INFRA"
  [ -f "$INFRA/docker-compose.yml" ] || die "infra/docker-compose.yml missing"

  print_banner "shutdown"

  if [ "$do_reset" -eq 1 ]; then
    _print_reset_warning
    if [ "$assume_yes" -ne 1 ]; then
      if [ "$IS_TTY" -eq 0 ]; then
        die "non-TTY and --yes not provided; refusing destructive reset"
      fi
      local reply
      read -r -p "  Proceed? Type 'yes' to continue: " reply
      [ "$reply" = "yes" ] || die "aborted"
    else
      printf '  %s(--yes assumed)%s\n' "$C_DIM" "$C_RESET"
    fi
    do_reset_teardown
    _shutdown_k8s delete
    printf '\n  %s✓ stack wiped%s — Compose containers, named volumes, project images (built + pulled), data/, and the minikube cluster removed.\n' \
      "$C_GREEN" "$C_RESET"
    printf '  run %s./mediwatch.sh init%s to bring it all back (will rebuild images).\n\n' \
      "$C_CYAN" "$C_RESET"
  else
    hr "docker compose stop"
    ( cd "$INFRA" && docker compose stop )
    _shutdown_k8s stop
    echo
    printf '  %s✓ containers halted%s — volumes, images, and data/ preserved, minikube cluster stopped.\n' \
      "$C_GREEN" "$C_RESET"
    printf '  run %s./mediwatch.sh init%s to resume.\n\n' \
      "$C_CYAN" "$C_RESET"
  fi
}

# ============================================================================
# cmd_run / cmd_train: run the DATA-PREP notebooks (01..05) on the host via
# the pipeline/run_pipeline.py wrapper. Thin shell over `python
# pipeline/run_pipeline.py "$@"`, with two prechecks: python must resolve on
# PATH (the caller activates the right env) and run_pipeline.py must exist.
# With no args we cap the range at notebook 05
# (--to 05_split_encode_scale_select.ipynb). HPO/training/registration (06..08)
# are owned by the airflow 'retrain_on_drift' DAG and run inside containers (no
# host-side Ray client). All explicit args are forwarded verbatim to
# run_pipeline.py, so its flags (--from, --to, --timeout, --reset, --keep-going,
# --list, positional names) work identically through this wrapper.
#
# `train` is the preferred name (paired with `activate` for serve-only flow).
# `run` is an alias, both dispatch here.
# ============================================================================

cmd_run_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh train [pipeline-args...] [--help]
       ./mediwatch.sh run   [pipeline-args...]        (alias for train)

Run the DATA-PREPARATION notebooks (01..05) on the host using the active
Python environment, by delegating to pipeline/run_pipeline.py. This produces
the frozen training reference (data/features.csv, data/train_test.npz) the
champion trains on.

Hyperparameter tuning, training, and champion registration (06..08) are NOT
run here — they are owned by the airflow 'retrain_on_drift' DAG, which runs
inside the containers (no host-side Ray client). After data prep, mint/refresh
the champion with:
  ./mediwatch.sh retrain      (HPO -> training -> conclusion -> champion gate)

To only SERVE an existing champion, use './mediwatch.sh activate' instead.

Prechecks (NOT the full audit — see './mediwatch.sh doctor' for that):
  · python must resolve on PATH (activate your env first)
  · pipeline/run_pipeline.py must be present

Behaviour:
  · With no args, runs 01..05 (caps at 05_split_encode_scale_select.ipynb).
  · Any explicit args are passed to run_pipeline.py verbatim (advanced use).

Common pipeline flags (passed through verbatim):
  --reset                 wipe data/ + clear notebook outputs before running
  --from <name>           start from this notebook (inclusive)
  --to <name>             stop at this notebook (inclusive)
  --timeout <seconds>     per-cell timeout (CI uses 10800 for the full HPO pass)
  --keep-going            don't stop on the first failure
  --list                  print discovered notebook order and exit
  <name> [<name>...]      positional: run exactly these notebooks, in order

Examples:
  ./mediwatch.sh train                              # data prep 01..05
  ./mediwatch.sh train --list                       # show discovered order
  ./mediwatch.sh train --from 04_feature_engineering.ipynb --to 05_split_encode_scale_select.ipynb
  ./mediwatch.sh train --reset --keep-going

For the underlying script's full flag list:
  python pipeline/run_pipeline.py --help
EOF
}

cmd_run() {
  local pipeline_args=()
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_run_help; return 0 ;;
      *)         pipeline_args+=("$arg") ;;
    esac
  done

  print_banner "train"

  # Lightweight prechecks: only what the pipeline needs.
  command -v python >/dev/null 2>&1 \
    || die "python not found on PATH — activate your environment (see './mediwatch.sh doctor')"
  [ -d "$ROOT/pipeline" ] \
    || die "pipeline/ directory not found at $ROOT/pipeline"
  [ -f "$ROOT/pipeline/run_pipeline.py" ] \
    || die "pipeline/run_pipeline.py missing at $ROOT/pipeline/run_pipeline.py"

  # Host runs DATA PREP ONLY (01..05). HPO/training/registration (06..08) belong
  # to the airflow 'retrain_on_drift' DAG (driven inside the containers, so the
  # host never opens a ray:// client). With no explicit selection we cap the
  # range at notebook 05. Any caller-supplied args are honoured verbatim. NB02-05
  # do not touch MLflow, so no MLflow precheck is needed here.
  if [ "${#pipeline_args[@]}" -eq 0 ]; then
    pipeline_args=(--to 05_split_encode_scale_select.ipynb)
  fi

  hr "python pipeline/run_pipeline.py ${pipeline_args[*]:-}"
  ( cd "$ROOT" && python pipeline/run_pipeline.py "${pipeline_args[@]}" )

  if [ $? -eq 0 ]; then
    echo
    info "data prep complete (01..05). Next, mint/refresh the champion via airflow:"
    info "  ./mediwatch.sh retrain     (HPO -> training -> conclusion -> champion gate)"
  fi
}

# ============================================================================
# cmd_retrain / cmd_drift: host-side triggers for the airflow orchestration
# plane. Both shell into the running airflow-scheduler container and fire a DAG
# via the airflow CLI. The host needs only Docker: no Python ML deps, no Ray
# client, no airflow client. This keeps HPO/training off the host and inside
# the containers (the airflow worker is the Ray driver).
# ============================================================================

# trigger_dag <dag-id> [<--conf json>]
trigger_dag() {
  local dag_id="$1" conf="${2:-}"
  command -v docker >/dev/null 2>&1 || die "docker not found on PATH — run './mediwatch.sh doctor'"
  docker version >/dev/null 2>&1 || die "docker daemon unreachable — start Docker / dockerd"
  [ -f "$INFRA/docker-compose.yml" ] || die "infra/docker-compose.yml missing"
  hr "docker compose exec airflow-scheduler airflow dags trigger $dag_id"
  if [ -n "$conf" ]; then
    ( cd "$INFRA" && docker compose exec -T airflow-scheduler airflow dags trigger "$dag_id" --conf "$conf" )
  else
    ( cd "$INFRA" && docker compose exec -T airflow-scheduler airflow dags trigger "$dag_id" )
  fi || die "failed to trigger '$dag_id'. Is the stack up? Run './mediwatch.sh init' (airflow at http://localhost:18080)."
}

cmd_retrain_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh retrain [--help]

Trigger the airflow 'retrain_on_drift' DAG: HPO (NB06) -> training (NB07) ->
conclusion + champion registration (NB08) -> champion gate -> inference reload.
Runs entirely in containers (the airflow worker is the Ray driver); the host
only fires the trigger. Use this to mint the initial champion after data prep
(./mediwatch.sh train), or to retrain by hand. Requires the stack to be up
('./mediwatch.sh init').

Watch progress in the Airflow UI at http://localhost:18080.
EOF
}

cmd_retrain() {
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_retrain_help; return 0 ;;
      *) echo "retrain: unknown argument: $arg" >&2
         echo "run './mediwatch.sh retrain --help' for usage." >&2
         exit 2 ;;
    esac
  done
  print_banner "retrain"
  trigger_dag "retrain_on_drift"
  echo
  printf '  %sretrain_on_drift triggered%s\n' "$C_GREEN" "$C_RESET"
  info "HPO -> training -> conclusion -> champion gate, then inference-api reload."
  info "Watch: http://localhost:18080  (Airflow UI)"
}

cmd_drift_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh drift [<scenario>] [--help]

Stage a simulated 'newly-arrived' production batch and trigger the airflow
'scheduled_drift_check' DAG against it. The DAG computes PSI/KS over the
monitored columns and, on an ALERT verdict, triggers 'retrain_on_drift'.

<scenario> is one of the batches generated by pipeline/09_drift_simulation.ipynb
into data/incoming/ (default: coding_shift):
  none  coding_shift  casemix_shift  los_utilization_shift  formulary_shift  mixed_severe

'none' demonstrates the no-retrain path; the others trip an ALERT and retrain.
Requires the stack up ('./mediwatch.sh init') and NB09 to have been run.
EOF
}

cmd_drift() {
  local scenario=""
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_drift_help; return 0 ;;
      -*) echo "drift: unknown argument: $arg" >&2
          echo "run './mediwatch.sh drift --help' for usage." >&2
          exit 2 ;;
      *) [ -z "$scenario" ] && scenario="$arg" ;;
    esac
  done
  [ -z "$scenario" ] && scenario="coding_shift"

  print_banner "drift"
  local src="$ROOT/data/incoming/$scenario.csv"
  local dst="$ROOT/data/incoming/current.csv"
  if [ ! -f "$src" ]; then
    die "scenario batch not found: $src
  Run pipeline/09_drift_simulation.ipynb first to generate data/incoming/*.csv"
  fi
  cp "$src" "$dst"
  info "staged data/incoming/current.csv  <-  $scenario.csv"
  trigger_dag "scheduled_drift_check"
  echo
  printf '  %sscheduled_drift_check triggered against '\''%s'\''%s\n' "$C_GREEN" "$scenario" "$C_RESET"
  info "On an ALERT verdict it auto-triggers retrain_on_drift."
  info "Watch: http://localhost:18080  (Airflow UI)"
}

# ============================================================================
# cmd_activate: serve-only path. Bring the inference-api up against a model
# that's already trained and registered in MLflow.
#
# Why this is separate from `init`:
#   init  → boots the FULL platform (postgres, airflow, ray, mlflow, prom,
#           grafana, inference-api). Needed for the retrain lifecycle.
#   train → runs the data-prep notebooks (01..05). retrain triggers the airflow
#           DAG that does HPO, eval, and registry promotion inside containers.
#   activate → boots ONLY postgres + mlflow + inference-api. The inference
#           container's compose `depends_on: { mlflow: service_healthy }`
#           pulls the dep chain automatically: we `compose up -d inference-api`
#           and let compose do the rest. No airflow, no ray, no observability
#           stack. ~2.5 GB resident vs ~8 GB for full init.
#
# Post-up check uses /healthz: 200 means a champion is loaded and the API is
# serving. 503 means MLflow is up but no `medi-watch-readmission@champion`
# alias exists yet, in which case the user needs to `train` first.
# ============================================================================

cmd_activate_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh activate [--help]

Bring up ONLY what's needed to serve the medi-watch inference API against an
already-trained `medi-watch-readmission@champion` model in MLflow.

What gets started (via 'docker compose up -d inference-api'):
  · postgres        (MLflow's metadata store; pulled in as a dep)
  · mlflow          (so the API can resolve @champion; pulled in as a dep)
  · inference-api   (the FastAPI scorer on http://localhost:18002)

What does NOT get started:
  · airflow, ray, prometheus, grafana  → use 'init' if you need them

Prechecks:
  · docker daemon reachable
  · infra/docker-compose.yml present

After services come up, polls /healthz for up to 60s:
  · 200 → champion is loaded; prints API URL + sample curl
  · 503 → MLflow is up but no @champion alias yet — run './mediwatch.sh train'

If you've never run 'init', this also serves as a first-time bootstrap for
the minimal serve-only footprint.

Examples:
  ./mediwatch.sh activate                # start serving the current champion
  curl http://localhost:18002/healthz    # verify externally
EOF
}

cmd_activate() {
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_activate_help; return 0 ;;
      *) echo "activate: unknown argument: $arg" >&2
         echo "run './mediwatch.sh activate --help' for usage." >&2
         exit 2 ;;
    esac
  done

  print_banner "activate"

  command -v docker >/dev/null 2>&1 \
    || die "docker not found on PATH — run './mediwatch.sh doctor' to install"
  docker version >/dev/null 2>&1 \
    || die "docker daemon unreachable — start dockerd (or check group membership)"
  [ -f "$INFRA/docker-compose.yml" ] \
    || die "$INFRA/docker-compose.yml missing"

  # GPU overlay selection (inference-api itself is CPU-only, but keep COMPOSE_FILE
  # consistent with whatever `init` brought up so depends_on services match).
  select_compose_files

  hr "docker compose up -d inference-api   (pulls postgres + mlflow via depends_on)"
  ( cd "$INFRA" && docker compose up -d inference-api )

  # Poll /healthz. The inference-api container's own healthcheck has a 25s
  # start_period + 5×15s retries, so we mirror that with a 60s ceiling.
  # 200 = champion loaded, 503 = MLflow up but no @champion alias yet.
  # We grab body unconditionally (not -f) so we can pattern-match the
  # "no champion" error and bail fast rather than waiting for the timeout.
  local url="http://localhost:18002/healthz"
  info "waiting for inference-api /healthz (up to 60s)..."
  local i code body tmp
  tmp=$(mktemp)
  for i in $(seq 1 30); do
    code=$(curl -s -o "$tmp" -w '%{http_code}' --max-time 2 "$url" 2>/dev/null || echo 000)
    body=$(cat "$tmp" 2>/dev/null || true)
    if [ "$code" = "200" ]; then
      rm -f "$tmp"
      printf '\n  %s✓ inference-api is serving%s\n' "$C_GREEN" "$C_RESET"
      printf '    %surl%s    http://localhost:18002\n' "$C_DIM" "$C_RESET"
      printf '    %shealth%s %s\n' "$C_DIM" "$C_RESET" "$body"
      printf '\n  %sTry it:%s\n' "$C_BOLD" "$C_RESET"
      printf '    curl http://localhost:18002/healthz\n'
      printf '    curl http://localhost:18002/version\n\n'
      return 0
    fi
    # Definitive "no champion" error → fail fast. The model cannot appear
    # without an explicit registration step, so further polling is pointless.
    case "$body" in
      *"RESOURCE_DOES_NOT_EXIST"*|*"medi-watch-readmission not found"*|*"medi-watch-readmission@champion"*)
        rm -f "$tmp"
        warn "MLflow is up but 'medi-watch-readmission@champion' is not registered."
        warn "Mint a champion first (fresh checkout):"
        warn "    ./mediwatch.sh train     # data prep, notebooks 01-05"
        warn "    ./mediwatch.sh retrain   # HPO + train + register @champion"
        die "inference-api cannot serve without a champion model"
        ;;
    esac
    sleep 2
  done
  rm -f "$tmp"

  warn "/healthz did not return 200 within 60s"
  warn "logs: ( cd $INFRA && docker compose logs --tail=50 inference-api )"
  die "inference-api failed to come healthy"
}

# ============================================================================
# cmd_k8s: apply infra/k8s manifests in deploy order to the current kubectl
# context. The four images (mlops-mlflow, mlops-ray, mlops-airflow,
# mlops-inference-api) come from `docker compose build`. --build runs that
# build, then loads them into a local kind/k3d cluster (detected from the
# kubectl context name) since neither cluster can pull from the host daemon.
# ============================================================================

K8S_NAMESPACE_DEFAULT="medi-watch"
# Order matters: airflow + inference-api both resolve mlflow.<ns> at startup,
# and ray workers join ray-head via GCS. Mirrors infra/k8s/README.md.
K8S_MANIFESTS=(
  "mlflow.yaml"
  "ray.yaml"
  "airflow.yaml"
  "inference-api.yaml"
)
K8S_IMAGES=(
  "mlops-mlflow:latest"
  "mlops-ray:latest"
  "mlops-airflow:latest"
  "mlops-inference-api:latest"
)

cmd_k8s_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh k8s [--start-cluster] [--build] [--reset] [--down]
                          [--status] [--namespace <ns>]
                          [--cluster auto|kind|k3d|none|minikube]
                          [--yes|-y] [--help]
       ./mediwatch.sh k8  [...]                          (alias for k8s)

On a LOCAL minikube cluster (context 'minikube', or with --start-cluster), this
deploys the inference API ONLY via infra/k8s/deploy-local.sh — matching the
current topology where k8s runs the inference service and airflow/ray/mlflow
stay in docker-compose.

On kind / k3d / external contexts it applies the legacy four-manifest stack in
deploy order (mlflow → ray → airflow → inference-api), each step blocking on
the workload becoming ready before the next is applied.

  --start-cluster One-command local bootstrap: `minikube start` (if not already
                  running), switch kubectl to the 'minikube' context, then build
                  + load + deploy the inference API. Needs a reachable MLflow
                  with a registered @champion (pods resolve it at startup).

Prechecks:
  · kubectl on PATH and a reachable cluster
  · infra/k8s/ directory present with the four manifests

Options:
  --build         Run 'docker compose build' for the four MLOps images, then
                  load them into the cluster's container runtime (kind: 'kind
                  load docker-image'; k3d: 'k3d image import'). Skipped for
                  external clusters where images come from a registry.
  --reset         Delete the namespace before applying. Wipes EVERYTHING in
                  it: Deployments, StatefulSets, PVCs (Postgres + artifacts +
                  Airflow logs), Secrets, Services. Prompts unless --yes.
  --down          Tear down only: delete the namespace and exit without
                  re-applying. Mutually exclusive with --build / --status.
  --status        Print pods, services, and PVCs in the namespace, then exit.
                  Read-only; skips apply.
  --namespace ns  Namespace to apply into. Default: medi-watch (matches the
                  manifests' hardcoded metadata.namespace).
  --cluster MODE  How to load images for --build:
                    auto  detect from kubectl context (kind-* or k3d-*)
                    kind  force 'kind load docker-image'
                    k3d   force 'k3d image import'
                    none  skip image loading (registry-backed cluster)
                  Default: auto.
  --yes, -y       Skip the --reset / --down confirmation prompt.

Environment:
  IMAGE_TAG       Tag substituted into the manifests' mlops-* image references
                  at apply time. Default: latest (the tag --build produces).
                  Set IMAGE_TAG=<tag> for a registry-backed cluster serving a
                  pinned build, e.g. IMAGE_TAG=v1.2.3 ./mediwatch.sh k8s.

Examples:
  ./mediwatch.sh k8s                       # apply to current context, wait for ready
  ./mediwatch.sh k8s --status              # what's running in the namespace
  ./mediwatch.sh k8s --build               # build + load images, then apply
  ./mediwatch.sh k8s --reset --build --yes # full clean redeploy, no prompts
  ./mediwatch.sh k8s --down --yes          # tear the namespace down

Service URLs (in-cluster DNS, resolved via kube-dns):
  mlflow:         mlflow.<ns>.svc.cluster.local:5000
  ray dashboard:  ray-head.<ns>.svc.cluster.local:8265
  airflow:        airflow.<ns>.svc.cluster.local:8080
  inference-api:  inference-api.<ns>.svc.cluster.local:80

For host access, port-forward:
  kubectl -n <ns> port-forward svc/mlflow 5000:5000
EOF
}

# _k8s_detect_cluster: echo "kind", "k3d", or "external" based on the
# current kubectl context name. kind contexts start with "kind-", k3d with
# "k3d-". Anything else is treated as external (no host-side image loading).
_k8s_detect_cluster() {
  local ctx
  ctx=$(kubectl config current-context 2>/dev/null || echo "")
  case "$ctx" in
    kind-*)    echo "kind" ;;
    k3d-*)     echo "k3d"  ;;
    minikube)  echo "minikube" ;;
    *)         echo "external" ;;
  esac
}

# _k8s_minikube_bootstrap: ensure a local minikube cluster is running and is the
# current kubectl context. Idempotent — a no-op when the apiserver is already up.
# Backs `./mediwatch.sh k8s --start-cluster`, the one-command "cluster up + deploy"
# path for local development.
_k8s_minikube_bootstrap() {
  command -v minikube >/dev/null 2>&1 \
    || die "minikube not found on PATH — install it (https://minikube.sigs.k8s.io/docs/start/) then re-run"
  command -v docker >/dev/null 2>&1 \
    || die "docker not found on PATH (minikube --driver=docker needs it)"

  local status
  status="$(minikube status 2>/dev/null || true)"
  if printf '%s' "$status" | grep -q "apiserver: Running"; then
    info "minikube already running"
  else
    # A cluster whose host container is Running but whose apiserver is NOT is
    # degraded (e.g. a crash-looping kubelet from a corrupted /etc/kubernetes).
    # `minikube start` will try to revive it in place and can block on apiserver
    # verification effectively forever — the freeze behind `init --k8s` hangs.
    # Warn loudly and time-box the start so a wedged cluster fails fast with
    # actionable guidance instead of hanging the whole init.
    if printf '%s' "$status" | grep -q "host: Running"; then
      warn "minikube container is up but the apiserver is down — cluster is degraded."
      warn "if start stalls, the cluster is likely corrupted; recover with:"
      warn "  minikube delete && ./mediwatch.sh init --k8s"
    fi
    hr "minikube start --driver=docker"
    # Prefer GNU timeout (coreutils) / gtimeout (macOS brew). Fall back to an
    # un-timed start only if neither is present, preserving prior behavior.
    local to_bin="" rc=0
    if command -v timeout >/dev/null 2>&1; then to_bin="timeout"
    elif command -v gtimeout >/dev/null 2>&1; then to_bin="gtimeout"; fi
    local start_timeout="${MINIKUBE_START_TIMEOUT:-420}"
    if [ -n "$to_bin" ]; then
      "$to_bin" "$start_timeout" \
        minikube start --driver=docker --cpus=2 --memory=4096 || rc=$?
    else
      minikube start --driver=docker --cpus=2 --memory=4096 || rc=$?
    fi
    if [ "$rc" -eq 124 ]; then
      die "minikube start timed out after ${start_timeout}s — the cluster is wedged (often a crash-looping kubelet). Recover with: minikube delete && ./mediwatch.sh init --k8s  (override the cap with MINIKUBE_START_TIMEOUT=<seconds>)"
    elif [ "$rc" -ne 0 ]; then
      die "minikube start failed (see output above)"
    fi
  fi
  # minikube start sets the context, but be explicit so a stale context from a
  # prior kind/k3d session does not silently misdirect the deploy.
  kubectl config use-context minikube >/dev/null 2>&1 \
    || die "could not switch kubectl context to 'minikube'"
}

# _k8s_deploy_minikube_local: deploy the inference API to minikube via the
# inference-only flow (infra/k8s/deploy-local.sh) — build + minikube image load +
# the four-variable envsubst the EKS-ready manifest needs. This deliberately
# bypasses the legacy four-manifest apply loop below: on k8s, medi-watch runs the
# inference API only (airflow/ray/mlflow live in docker-compose).
_k8s_deploy_minikube_local() {
  local script="$INFRA/k8s/deploy-local.sh"
  [ -f "$script" ] || die "missing $script (the local inference-only deploy)"

  # deploy-local.sh resolves @champion from MLflow at pod startup; surface that
  # dependency rather than letting the rollout silently time out on a 503.
  local mlflow_uri="${MLFLOW_TRACKING_URI:-http://host.minikube.internal:5000}"
  info "mlflow:    $mlflow_uri  (must be reachable from the cluster, with a registered @champion)"
  warn "inference pods need a reachable MLflow + medi-watch-readmission@champion, or /healthz stays 503"

  hr "infra/k8s/deploy-local.sh  (CLUSTER_TOOL=minikube)"
  CLUSTER_TOOL=minikube \
  MLFLOW_TRACKING_URI="$mlflow_uri" \
  K8S_NAMESPACE="$K8S_NAMESPACE_DEFAULT" \
  ROLLOUT_WAIT="${ROLLOUT_WAIT:-true}" \
    bash "$script"
}

# _k8s_build_and_load: build the four MLOps images via docker compose, then
# import them into the cluster's runtime. Loader is selected per cluster mode.
_k8s_build_and_load() {
  local cluster_mode="$1"

  hr "docker compose build (mlflow, ray, airflow, inference-api)"
  ( cd "$INFRA" && docker compose build mlflow ray airflow inference-api )

  if [ "$cluster_mode" = "none" ] || [ "$cluster_mode" = "external" ]; then
    info "cluster mode '$cluster_mode' — skipping local image load"
    info "push these images to a registry the cluster can pull from:"
    local img
    for img in "${K8S_IMAGES[@]}"; do
      printf '    %s\n' "$img"
    done
    return 0
  fi

  hr "load images into the $cluster_mode cluster"
  local ctx cluster_name img
  ctx=$(kubectl config current-context)
  # context name = "<tool>-<cluster>". The cluster name is what kind/k3d want.
  cluster_name="${ctx#${cluster_mode}-}"

  case "$cluster_mode" in
    kind)
      command -v kind >/dev/null 2>&1 \
        || die "kind not found on PATH (needed to load images into kind cluster '$cluster_name')"
      for img in "${K8S_IMAGES[@]}"; do
        info "kind load docker-image $img --name $cluster_name"
        kind load docker-image "$img" --name "$cluster_name"
      done
      ;;
    k3d)
      command -v k3d >/dev/null 2>&1 \
        || die "k3d not found on PATH (needed to load images into k3d cluster '$cluster_name')"
      for img in "${K8S_IMAGES[@]}"; do
        info "k3d image import $img -c $cluster_name"
        k3d image import "$img" -c "$cluster_name"
      done
      ;;
  esac
}

# _k8s_delete_namespace: best-effort namespace teardown. `kubectl delete ns`
# cascades to every resource inside the namespace, then we wait for it to
# fully terminate so the subsequent apply doesn't race a still-deleting ns.
_k8s_delete_namespace() {
  local ns="$1"
  if ! kubectl get namespace "$ns" >/dev/null 2>&1; then
    info "namespace '$ns' does not exist — nothing to delete"
    return 0
  fi
  info "kubectl delete namespace $ns --wait=true (cascades to pods/pvcs/secrets)"
  kubectl delete namespace "$ns" --wait=true --timeout=180s \
    || warn "namespace delete returned non-zero; continuing"

  # Belt-and-suspenders: poll until the namespace is gone. The previous
  # `--wait=true` usually suffices, but stuck finalizers can leave it in
  # 'Terminating' state past the timeout.
  local i
  for i in $(seq 1 60); do
    kubectl get namespace "$ns" >/dev/null 2>&1 || return 0
    sleep 2
  done
  die "namespace '$ns' still present after 120s — investigate stuck finalizers (kubectl get ns $ns -o yaml)"
}

# _k8s_wait_for: wait for the primary workload of one manifest to reach
# ready/available. We hard-code the kind+name pairs rather than parsing YAML,
# the manifest set is small and stable (see infra/k8s/README.md).
_k8s_wait_for() {
  local ns="$1" manifest="$2"
  case "$manifest" in
    mlflow.yaml)
      info "wait: statefulset/mlflow-postgres (Postgres) + deployment/mlflow"
      kubectl -n "$ns" rollout status statefulset/mlflow-postgres --timeout=300s
      kubectl -n "$ns" rollout status deployment/mlflow            --timeout=300s
      ;;
    ray.yaml)
      info "wait: deployment/ray-head + deployment/ray-worker"
      kubectl -n "$ns" rollout status deployment/ray-head   --timeout=300s
      kubectl -n "$ns" rollout status deployment/ray-worker --timeout=300s
      ;;
    airflow.yaml)
      info "wait: deployment/airflow"
      kubectl -n "$ns" rollout status deployment/airflow --timeout=300s
      ;;
    inference-api.yaml)
      info "wait: deployment/inference-api"
      kubectl -n "$ns" rollout status deployment/medi-watch-inference --timeout=300s
      ;;
    *)
      warn "no wait rule for $manifest — skipping readiness check"
      ;;
  esac
}

cmd_k8s() {
  local do_build=0 do_reset=0 do_down=0 do_status=0 assume_yes=0 do_start_cluster=0
  local namespace="$K8S_NAMESPACE_DEFAULT"
  local cluster_mode="auto"

  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)    cmd_k8s_help; return 0 ;;
      --build)      do_build=1 ;;
      --reset)      do_reset=1 ;;
      --down)       do_down=1 ;;
      --status)     do_status=1 ;;
      --start-cluster) do_start_cluster=1 ;;
      --yes|-y)     assume_yes=1 ;;
      --namespace)  shift; namespace="${1:-}";    [ -z "$namespace" ]    && die "--namespace requires a value" ;;
      --cluster)    shift; cluster_mode="${1:-}"; [ -z "$cluster_mode" ] && die "--cluster requires a value" ;;
      *) echo "k8s: unknown argument: $1" >&2
         echo "run './mediwatch.sh k8s --help' for usage." >&2
         exit 2 ;;
    esac
    shift
  done

  # --start-cluster bootstraps a local minikube cluster and deploys the
  # inference API to it (the only medi-watch k8s workload), so it implies the
  # minikube cluster mode.
  if [ "$do_start_cluster" -eq 1 ]; then
    cluster_mode="minikube"
  fi

  case "$cluster_mode" in
    auto|kind|k3d|none|external|minikube) ;;
    *) die "invalid --cluster '$cluster_mode' (expected: auto|kind|k3d|none|minikube)" ;;
  esac

  if [ "$do_down" -eq 1 ] && { [ "$do_build" -eq 1 ] || [ "$do_status" -eq 1 ]; }; then
    die "--down is mutually exclusive with --build / --status"
  fi

  command -v kubectl >/dev/null 2>&1 \
    || die "kubectl not found on PATH — install kubectl, then 'kubectl config use-context <your-cluster>'"
  [ -d "$INFRA/k8s" ] \
    || die "infra/k8s/ directory not found at $INFRA/k8s"
  local m
  for m in "${K8S_MANIFESTS[@]}"; do
    [ -f "$INFRA/k8s/$m" ] || die "missing manifest: $INFRA/k8s/$m"
  done

  print_banner "k8s"

  # --start-cluster: bring a local minikube cluster up first so the reachability
  # check below passes and the deploy targets it.
  if [ "$do_start_cluster" -eq 1 ]; then
    _k8s_minikube_bootstrap
  fi

  # Cluster reachability: fail fast with a clear hint instead of letting
  # downstream kubectl calls error one by one.
  local ctx
  ctx=$(kubectl config current-context 2>/dev/null || echo "")
  [ -n "$ctx" ] || die "no kubectl current-context set (run: kubectl config get-contexts)"
  if ! kubectl cluster-info >/dev/null 2>&1; then
    die "kubectl context '$ctx' is set but the cluster is unreachable (start it or switch contexts)"
  fi

  # Resolve auto → concrete mode using the context name.
  if [ "$cluster_mode" = "auto" ]; then
    cluster_mode=$(_k8s_detect_cluster)
  fi

  info "context:   $ctx"
  info "namespace: $namespace"
  info "cluster:   $cluster_mode"

  # --status: read-only inventory, then exit.
  if [ "$do_status" -eq 1 ]; then
    hr "kubectl get pods,svc,pvc -n $namespace"
    if ! kubectl get namespace "$namespace" >/dev/null 2>&1; then
      warn "namespace '$namespace' does not exist — run './mediwatch.sh k8s' to deploy"
      return 0
    fi
    kubectl get pods,svc,pvc -n "$namespace"
    return 0
  fi

  # --down: delete and exit (no apply).
  if [ "$do_down" -eq 1 ]; then
    if [ "$assume_yes" -ne 1 ]; then
      printf '  %s%s--down%s will delete namespace %s%s%s and ALL data in it\n' \
        "$C_AMBER" "$C_BOLD" "$C_RESET$C_AMBER" "$C_BOLD" "$namespace" "$C_RESET"
      printf '  (mlflow-postgres PVC, mlflow artifacts PVC, airflow PVC, secrets).\n\n'
      if [ "$IS_TTY" -eq 0 ]; then
        die "non-TTY and --yes not provided; refusing destructive delete"
      fi
      local reply
      read -r -p "  Proceed? Type 'yes' to continue: " reply
      [ "$reply" = "yes" ] || die "aborted"
    else
      printf '  %s(--yes assumed)%s\n' "$C_DIM" "$C_RESET"
    fi
    hr "delete namespace $namespace"
    _k8s_delete_namespace "$namespace"
    echo
    printf '  %s✓ namespace deleted%s — run %s./mediwatch.sh k8s%s to redeploy.\n\n' \
      "$C_GREEN" "$C_RESET" "$C_CYAN" "$C_RESET"
    return 0
  fi

  # --reset: same destructive flow, but then continue into apply.
  if [ "$do_reset" -eq 1 ]; then
    if [ "$assume_yes" -ne 1 ]; then
      printf '  %s%s--reset%s will delete namespace %s%s%s before re-applying\n' \
        "$C_AMBER" "$C_BOLD" "$C_RESET$C_AMBER" "$C_BOLD" "$namespace" "$C_RESET"
      printf '  (wipes Postgres + artifacts + airflow logs + secrets).\n\n'
      if [ "$IS_TTY" -eq 0 ]; then
        die "non-TTY and --yes not provided; refusing destructive reset"
      fi
      local reply
      read -r -p "  Proceed? Type 'yes' to continue: " reply
      [ "$reply" = "yes" ] || die "aborted"
    else
      printf '  %s(--yes assumed)%s\n' "$C_DIM" "$C_RESET"
    fi
    hr "delete namespace $namespace"
    _k8s_delete_namespace "$namespace"
  fi

  # minikube: deploy the inference API ONLY, via the inference-only local flow
  # (build + minikube image load + four-var envsubst). Returns before the legacy
  # four-manifest apply loop, which targets the old all-in-cluster topology.
  if [ "$cluster_mode" = "minikube" ]; then
    _k8s_deploy_minikube_local "$namespace"
    return 0
  fi

  # --build: docker compose build + load into the cluster's runtime.
  if [ "$do_build" -eq 1 ]; then
    command -v docker >/dev/null 2>&1 \
      || die "docker not found on PATH (needed by --build)"
    _k8s_build_and_load "$cluster_mode"
  fi

  # The manifests reference the four MLOps images as mlops-*:${IMAGE_TAG}.
  # kubectl does no variable substitution, so we expand that single placeholder
  # at apply time. envsubst is restricted to ${IMAGE_TAG} so any other ${...} a
  # manifest may later grow is passed through untouched. Default is :latest,
  # the tag --build produces and loads. Override with IMAGE_TAG=<tag> for a
  # registry-backed cluster pulling a pinned build (see --cluster none).
  command -v envsubst >/dev/null 2>&1 \
    || die "envsubst not found on PATH (ships with gettext) — needed to expand \${IMAGE_TAG} in the manifests"
  local image_tag="${IMAGE_TAG:-latest}"
  info "image tag: $image_tag"

  # Sequential apply, blocking on rollout between steps so each layer sees
  # its dependencies healthy when it boots. Bulk `apply -f .` also works,
  # but the inference-api Pod then restarts once while mlflow becomes
  # routable, so sequential is quieter on a fresh cluster.
  local rel
  for rel in "${K8S_MANIFESTS[@]}"; do
    hr "kubectl apply -f infra/k8s/$rel  (IMAGE_TAG=$image_tag)"
    IMAGE_TAG="$image_tag" envsubst '${IMAGE_TAG}' < "$INFRA/k8s/$rel" | kubectl apply -f -
    _k8s_wait_for "$namespace" "$rel"
  done

  echo
  hr "kubectl get pods -n $namespace"
  kubectl get pods -n "$namespace"

  echo
  printf '  %s✓ k8s stack is up%s in namespace %s%s%s\n' \
    "$C_GREEN" "$C_RESET" "$C_CYAN" "$namespace" "$C_RESET"
  printf '  Port-forward examples:\n'
  printf '    kubectl -n %s port-forward svc/mlflow        5000:5000\n' "$namespace"
  printf '    kubectl -n %s port-forward svc/ray-head      8265:8265\n' "$namespace"
  printf '    kubectl -n %s port-forward svc/airflow       8080:8080\n' "$namespace"
  printf '    kubectl -n %s port-forward svc/inference-api 8002:80\n\n'  "$namespace"
}

# ============================================================================
# cmd_install: register a 'medi-watch' shim on PATH so commands work from
# any directory. The shim is a 3-line wrapper that exec's the real
# mediwatch.sh by absolute path: no PATH gymnastics, no symlink-resolution
# logic needed in the script itself. Two canonical install paths:
#   /usr/local/bin/medi-watch    (default, system-wide, sudo)
#   $HOME/.local/bin/medi-watch  (--user, no sudo)
# Re-running install on either updates the wrapper to point at the current
# checkout. --uninstall removes the shim from BOTH paths.
# ============================================================================

# Generate the expected wrapper text for a given mediwatch.sh root directory.
# Used both for writing the shim and for the idempotency comparison.
_install_wrapper_content() {
  local src_root="$1"
  cat <<EOF
#!/usr/bin/env bash
# Auto-generated by mediwatch.sh install. Re-run to update.
exec "$src_root/mediwatch.sh" "\$@"
EOF
}

# Classify the state of an existing target file relative to the expected
# wrapper content. Echoes one of: fresh | match | conflict.
# Compares byte-for-byte via cmp -s. Both sides use the same trailing-newline
# convention (`printf '%s\n'` here and at write time), so the match is exact.
_install_check_existing() {
  local target="$1" expected="$2"
  if [ ! -e "$target" ]; then
    echo "fresh"; return
  fi
  if [ ! -r "$target" ]; then
    echo "conflict"; return
  fi
  if printf '%s\n' "$expected" | cmp -s - "$target"; then
    echo "match"
  else
    echo "conflict"
  fi
}

# After a --user install, warn if ~/.local/bin is not on PATH and emit a
# shell-specific one-liner to add it. System installs always land in
# /usr/local/bin which is on PATH by default on Linux and macOS.
_install_path_advisory() {
  local bin_dir="$HOME/.local/bin"
  case ":$PATH:" in
    *":$bin_dir:"*) return 0 ;;
  esac
  printf '\n  %s» %s is not on your PATH.%s\n' "$C_AMBER" "$bin_dir" "$C_RESET"
  local shell_name="${SHELL##*/}"
  case "$shell_name" in
    bash)
      printf "  add it with:\n    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc\n"
      printf '  then open a new shell, or run: source ~/.bashrc\n'
      ;;
    zsh)
      printf "  add it with:\n    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc\n"
      printf '  then open a new shell, or run: source ~/.zshrc\n'
      ;;
    *)
      printf '  add %s to PATH in your shell rc file, then open a new shell.\n' "$bin_dir"
      ;;
  esac
}

cmd_install_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh install [--user] [--uninstall] [--yes|-y] [--help]

Register a 'medi-watch' command on PATH so the platform's subcommands work
from any directory. The shim is a 3-line bash wrapper that exec's the real
mediwatch.sh inside this checkout. Re-run install to retarget the shim at
a moved or renamed checkout.

Options:
  --user       Install to $HOME/.local/bin/medi-watch (no sudo).
               Default target is /usr/local/bin/medi-watch (sudo).
               If ~/.local/bin is not on PATH, install prints a shell-
               specific one-liner to add it.
  --uninstall  Remove the shim from BOTH /usr/local/bin/medi-watch and
               $HOME/.local/bin/medi-watch if present. Silent-success
               when nothing is installed.
  --yes, -y    Skip the sudo confirmation prompt (system install) and the
               non-TTY refusal. Has no effect on the conflict refusal.

Idempotency:
  · If the shim already exists at the chosen path and points at THIS
    checkout, install reports 'already installed' and exits 0.
  · If a different file or a shim pointing elsewhere exists at the path,
    install refuses with a hint to run 'install --uninstall' first. --yes
    does NOT override this. Manual removal does.

Examples:
  ./mediwatch.sh install              # system-wide install (sudo)
  ./mediwatch.sh install --user       # user-local install (no sudo)
  ./mediwatch.sh install --uninstall  # remove from both canonical paths
  ./mediwatch.sh install --yes        # CI-style, skip sudo prompt
EOF
}

cmd_install() {
  local target_user=0 do_uninstall=0 assume_yes=0
  for arg in "$@"; do
    case "$arg" in
      -h|--help)   cmd_install_help; return 0 ;;
      --user)      target_user=1 ;;
      --uninstall) do_uninstall=1 ;;
      --yes|-y)    assume_yes=1 ;;
      *) echo "install: unknown argument: $arg" >&2
         echo "run './mediwatch.sh install --help' for usage." >&2
         exit 2 ;;
    esac
  done

  local system_path="/usr/local/bin/medi-watch"
  local user_path="$HOME/.local/bin/medi-watch"

  print_banner "install"

  if [ "$do_uninstall" -eq 1 ]; then
    hr "remove medi-watch shim from canonical paths"
    local removed=0 path
    for path in "$system_path" "$user_path"; do
      [ -e "$path" ] || continue
      if [ -w "$path" ] && [ -w "$(dirname "$path")" ]; then
        rm -f "$path"
      else
        sudo rm -f "$path"
      fi
      info "removed $path"
      removed=1
    done
    if [ "$removed" -eq 0 ]; then
      info "no medi-watch shim found (already uninstalled)"
    fi
    echo
    printf '  %s✓ uninstall complete%s\n\n' "$C_GREEN" "$C_RESET"
    return 0
  fi

  local target use_sudo
  if [ "$target_user" -eq 1 ]; then
    target="$user_path"
    use_sudo=0
  else
    target="$system_path"
    use_sudo=1
  fi

  local expected
  expected=$(_install_wrapper_content "$ROOT")

  local state
  state=$(_install_check_existing "$target" "$expected")

  case "$state" in
    match)
      hr "install"
      info "shim already installed at $target (points at this checkout)"
      ;;
    conflict)
      die "found existing medi-watch at $target that does not point at this checkout — remove it or run './mediwatch.sh install --uninstall' first."
      ;;
    fresh)
      hr "install"
      if [ "$use_sudo" -eq 1 ]; then
        info "writing $target (sudo required)"
        if [ "$assume_yes" -ne 1 ]; then
          if [ "$IS_TTY" -eq 0 ]; then
            die "non-TTY and --yes not provided; refusing to invoke sudo"
          fi
          local reply
          read -r -p "  Proceed with sudo write? Type 'yes' to continue: " reply
          [ "$reply" = "yes" ] || die "aborted"
        fi
        sudo mkdir -p "$(dirname "$target")"
        printf '%s\n' "$expected" | sudo tee "$target" >/dev/null
        sudo chmod 0755 "$target"
      else
        info "writing $target"
        mkdir -p "$(dirname "$target")"
        printf '%s\n' "$expected" > "$target"
        chmod 0755 "$target"
      fi
      ;;
  esac

  hr "verify"
  local resolved
  resolved=$(command -v medi-watch 2>/dev/null || true)
  if [ -z "$resolved" ]; then
    row badge_warn "command -v medi-watch" "not on PATH yet"
    note "open a new shell, or run: hash -r"
  else
    row badge_ok "command -v medi-watch" "$resolved"
  fi

  set +e
  medi-watch help >/dev/null 2>&1
  local rc=$?
  set -e
  if [ -z "$resolved" ]; then
    note "medi-watch help: skipped (shim not on PATH yet)"
  elif [ "$rc" -eq 0 ]; then
    row badge_ok "medi-watch help" "exit 0"
  else
    row badge_warn "medi-watch help" "exit $rc"
  fi

  if [ "$target_user" -eq 1 ]; then
    _install_path_advisory
  fi

  echo
  printf '  %s✓ install complete%s — run %smedi-watch help%s from anywhere.\n\n' \
    "$C_GREEN" "$C_RESET" "$C_CYAN" "$C_RESET"
}

# ============================================================================
# CLI dispatch
# ============================================================================

print_help() {
  cat <<'EOF'
mediwatch-cli — operations CLI for the medi-watch platform

Usage:
  ./mediwatch.sh <command> [options]

Commands:
  doctor    Audit + offer to install missing prerequisites (consent-prompted)
  init      Install + bring docker compose services up (full stack: airflow, mlflow,
            ray, prometheus, grafana, inference-api; preflight audit + optional --reset)
  install   Register a 'medi-watch' shim on PATH so commands work from any directory
  train     Run the DATA-PREP notebooks (01..05) on the host. (Alias: 'run'.)
  retrain   Trigger the airflow 'retrain_on_drift' DAG (HPO+train+register, NB06..08)
            in containers — the host only fires the trigger (no Ray client).
  drift     Stage a simulated batch + trigger 'scheduled_drift_check' (auto-retrains
            on an ALERT verdict).
  activate  Serve-only: bring up ONLY postgres + mlflow + inference-api against an
            already-trained champion. Use this when the model exists and you just
            want to host predictions — no airflow/ray/observability stack needed.
  k8s       Apply infra/k8s manifests (mlflow → ray → airflow → inference-api) to
            the current kubectl context. (Alias: 'k8'.)
  shutdown  Halt containers (--reset wipes volumes, ALL project images, and data/)
  help      Show this help

Typical lifecycles:
  Full retrain:   init  →  train  →  retrain  (data prep on host, then HPO+train
                                               +register via airflow in containers)
  Serve only:     activate                    (minimal footprint, existing champion)

Run './mediwatch.sh <command> --help' for command-specific help.
EOF
}

# ============================================================================
# cmd_doctor: interactive autofix without booting compose.
#
# Sequence: audit → offer_installers → re-audit. Useful on a fresh host where
# the user wants to converge prerequisites before any 'init'. Unlike init,
# this command never starts containers or touches data.
# ============================================================================

cmd_doctor_help() {
  cat <<'EOF'
Usage: ./mediwatch.sh doctor [--yes|-y] [--no-fix] [--help]

Single command for environment diagnosis and repair.

Always starts with a full read-only audit (host prereqs, Python env drift,
docker services, and infra/.env validation). Then:

  • Without --no-fix (default): offers interactive autofix for anything
    actionable, then re-audits.
  • With --no-fix: stops after the first audit report (pure read-only mode,
    equivalent to the old standalone 'audit' command). Exits non-zero if
    any WARN or MISS rows were present.

Installable categories (auto, prompts before running):
  host_docker, host_compose, host_git,
  py_medi_watch_drift, config_env

Manual-only categories (listed with hint, never auto-run):
  host_nvdrv, host_cuda, host_nvctk, host_python, py_medi_watch,
  config_env_core

Informational categories (visible in audit; doctor cannot resolve):
  docs_drift           INSTALL.md / .env.example mismatch (advisory only)
  docker_relogin       docker just installed; open a new shell to pick up
                       the 'docker' group, then re-run doctor
  docker_unreachable   docker on PATH but daemon won't respond (down or
                       permission denied)

Options:
  --yes, -y    Select every installable fix without prompting.
  --no-fix     Read-only audit only. Skip all autofix prompts and the
               re-audit step. Behaves like the former 'audit' command.

Examples:
  ./mediwatch.sh doctor               # full audit + interactive autofix
  ./mediwatch.sh doctor --no-fix      # read-only report (old "audit" behavior)
  ./mediwatch.sh doctor --yes         # full audit + auto-accept every fix
EOF
}

cmd_doctor() {
  local assume_yes=0
  local no_fix=0
  for arg in "$@"; do
    case "$arg" in
      -h|--help) cmd_doctor_help; return 0 ;;
      --yes|-y)  assume_yes=1 ;;
      --no-fix)  no_fix=1 ;;
      *) echo "doctor: unknown argument: $arg" >&2
         echo "run './mediwatch.sh doctor --help' for usage." >&2
         exit 2 ;;
    esac
  done

  print_banner "doctor"

  INSTALL_ASSUME_YES=$assume_yes
  INSTALL_DISABLED=$no_fix

  hr_section "$C_CYAN" "AUDIT"
  run_audit || true

  if [ "$MISS_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    printf '\n  %s✓ all clear%s — nothing to fix.\n\n' \
      "$C_GREEN" "$C_RESET"
    return 0
  fi

  if [ "$no_fix" -eq 1 ]; then
    # Read-only mode: report only, apply nothing.
    # Exit code matches what run_audit already set (non-zero on any MISS/WARN).
    printf '\n  %s» %d issue(s) found%s — re-run without --no-fix to offer autofixes.\n\n' \
      "$C_AMBER" "$((MISS_COUNT + WARN_COUNT))" "$C_RESET"
    return 1
  fi

  # Full doctor path: audit + consent-driven autofix + re-audit.
  offer_installers

  hr_section "$C_CYAN" "RE-AUDIT"
  run_audit || true

  if [ "$MISS_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
    printf '\n  %s✓ all clear%s — run %s./mediwatch.sh init%s to bring the stack up.\n\n' \
      "$C_GREEN" "$C_RESET" "$C_CYAN" "$C_RESET"
    return 0
  fi

  # Bucket remaining findings into three groups:
  #   surfaced       doctor has an installer or a manual hint for these
  #                  (the rows under "Needs manual attention" above).
  #   init_handled   no doctor installer, but `init` resolves them as a
  #                  side effect of `docker compose up -d` (docker_images).
  #   informational  advisory only, or resolvable solely outside doctor
  #                  (docs_drift, docker_relogin, docker_unreachable).
  # The split lets doctor pick the right closing message instead of telling
  # the user "init will handle it" when init genuinely cannot.
  local f state name detail cat meta surfaced=0 init_handled=0 informational=0
  for f in "${FINDINGS[@]:-}"; do
    [ -z "$f" ] && continue
    IFS=$'\t' read -r state name detail cat <<<"$f"
    case "$state" in miss|warn) ;; *) continue ;; esac
    case "$cat" in
      docs_drift|docker_relogin|docker_unreachable)
        informational=$((informational+1)); continue ;;
    esac
    meta=$(installer_for "$cat")
    if [ -n "$meta" ]; then
      surfaced=$((surfaced+1))
    else
      init_handled=$((init_handled+1))
    fi
  done

  if [ "$surfaced" -eq 0 ] && [ "$init_handled" -gt 0 ]; then
    printf '\n  %s» %d remaining issue(s) are handled by%s %s./mediwatch.sh init%s\n' \
      "$C_CYAN" "$init_handled" "$C_RESET" "$C_BOLD" "$C_RESET"
    printf '    (services not yet built — init runs %sdocker compose up -d%s to build them).\n\n' \
      "$C_DIM" "$C_RESET"
    return 0
  fi

  if [ "$surfaced" -eq 0 ] && [ "$init_handled" -eq 0 ] && [ "$informational" -gt 0 ]; then
    printf '\n  %s» %d informational note(s) above%s — see the rows tagged %sMISS%s/%sWARN%s.\n' \
      "$C_CYAN" "$informational" "$C_RESET" "$C_RED" "$C_RESET" "$C_AMBER" "$C_RESET"
    printf '    Nothing actionable for doctor here — follow the note next to each row.\n\n'
    return 0
  fi

  printf '\n  %s» %d issue(s) remain%s — see "Needs manual attention" above.\n\n' \
    "$C_AMBER" "$surfaced" "$C_RESET"
  return 1
}

# entry
#
# No-args path: print help, then (on TTY only) offer to run the full
# end-to-end lifecycle: init -> train -> retrain. For a read-only report, use
# `doctor --no-fix`. cmd_init runs preflight_audit internally as its gate.
# 'train' does host-side data prep (01..05). 'retrain' fires the airflow DAG
# that runs HPO/train/register (06..08) inside containers, so the champion is
# minted asynchronously: watch the Airflow UI. On a non-TTY (CI, pipes) we
# print help and exit.

if [ $# -eq 0 ]; then
  print_help

  if [ "$IS_TTY" -ne 1 ]; then
    exit 0
  fi

  printf '\n  %sNo command supplied.%s Run the full lifecycle (' \
    "$C_BOLD" "$C_RESET"
  printf '%sinit%s -> %strain%s -> %sretrain%s)?\n' \
    "$C_CYAN" "$C_RESET" "$C_CYAN" "$C_RESET" "$C_CYAN" "$C_RESET"
  printf '    %s1.%s init      bring docker compose services up (audit runs as preflight)\n' "$C_DIM" "$C_RESET"
  printf '    %s2.%s train     run the data-prep notebooks 01..05 on the host\n'             "$C_DIM" "$C_RESET"
  printf '    %s3.%s retrain   trigger the airflow HPO->train->register DAG (in containers)\n' "$C_DIM" "$C_RESET"
  read -r -p "  Type 'yes' to continue, anything else to exit: " LIFECYCLE_REPLY
  [ "$LIFECYCLE_REPLY" = "yes" ] || exit 0

  # cmd_init and cmd_run propagate failure via set -e, the correct behavior
  # for the lifecycle. cmd_init's own preflight_audit is the gate that aborts
  # on MISS findings. cmd_retrain only FIRES the DAG, training then proceeds
  # asynchronously inside airflow.
  cmd_init
  cmd_run
  cmd_retrain

  printf '\n  %s✓ lifecycle started%s — init + data prep done; retrain triggered.\n' \
    "$C_GREEN" "$C_RESET"
  printf '  The champion is minted asynchronously by airflow — watch %shttp://localhost:18080%s\n\n' \
    "$C_CYAN" "$C_RESET"
  exit 0
fi

CMD="$1"; shift
case "$CMD" in
  doctor)             cmd_doctor   "$@" ;;
  init)               cmd_init     "$@" ;;
  install)            cmd_install  "$@" ;;
  train)              cmd_run      "$@" ;;
  run)                cmd_run      "$@" ;;
  retrain)            cmd_retrain  "$@" ;;
  drift)              cmd_drift    "$@" ;;
  activate)           cmd_activate "$@" ;;
  k8s|k8)             cmd_k8s      "$@" ;;
  shutdown)           cmd_shutdown "$@" ;;
  help|-h|--help)     print_help ;;
  *)
    echo "Unknown command: $CMD" >&2
    echo "Run './mediwatch.sh help' for usage." >&2
    exit 2
    ;;
esac
