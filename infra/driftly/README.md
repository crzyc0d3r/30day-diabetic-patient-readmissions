# Driftly: model drift dashboard

A standalone **React + FastAPI** dashboard that computes and visualizes
distribution drift, on demand, between the champion model's training reference
and a chosen current batch. Three metrics per monitored feature:

- **Wasserstein distance** (normalized by reference std): numeric features.
- **Population Stability Index (PSI)**: reuses `helpers/drift_sim.py`.
- **Kullback–Leibler (KL) divergence**: `KL(current ‖ reference)`.

It complements the existing Airflow drift pipeline and Grafana. **No MLflow
dependency.** Styled to match the medi-watch `README.html` design system
(dark/light dual theme, IBM Plex, blue accent).

## Run it

From `infra/`, with the stack's data materialised (`features.csv` present):

```bash
docker compose up -d driftly-api driftly-web
```

- Dashboard: <http://localhost:3004>
- API (direct): <http://localhost:8003/api/health>, docs at `/docs`

`driftly-web` (nginx) reverse-proxies `/api` to `driftly-api`, so the browser
makes same-origin calls.

## Data it reads

| Role | Path |
|---|---|
| Reference | `data/features.csv` (NB04 engineered matrix; champion's cohort) |
| Current batches | `data/incoming/*.csv` (NB09 scenarios; `current.csv` staging) |
| Upload | any CSV with the monitored columns |
| History | `data/driftly/history.db` (SQLite, created on first compute) |

Monitored columns and PSI thresholds come from `helpers/constants.py`, so
Driftly and the Airflow drift check agree on what "drift" means.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/batches` | available current batches |
| GET | `/api/features` | monitored columns + type |
| POST | `/api/compute` | compute drift for `{batch}` (JSON) |
| POST | `/api/compute/upload` | compute drift for an uploaded CSV (multipart) |
| GET | `/api/history` | past runs for the trend (`?feature=` for a per-feature series) |

## Thresholds

PSI reuses `DRIFT_PSI_WARN`/`DRIFT_PSI_ALERT` (0.1 / 0.2). Wasserstein and KL
have their own env-overridable bands (defaults in `backend/config.py`):

```
DRIFTLY_WASSERSTEIN_WARN=0.10   DRIFTLY_WASSERSTEIN_ALERT=0.25
DRIFTLY_KL_WARN=0.10            DRIFTLY_KL_ALERT=0.25
```

## Layout

```
infra/driftly/
  backend/   FastAPI: main.py, drift_metrics.py, history.py, config.py, tests/
  frontend/  React+Vite+TS+Recharts: src/{api,theme,types}.ts, components/, styles/
```

## Develop / test

```bash
# backend (ephemeral env)
cd backend
uv run --no-project --python 3.13 --with numpy --with pandas --with scipy \
  --with fastapi --with httpx --with python-multipart --with pytest -- \
  python -m pytest tests/ -o addopts=""

# frontend
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api to :8003)
npm test           # vitest
```

The Wasserstein/KL math lives in the shared `helpers/drift_sim.py` (alongside
the existing PSI/KS), covered by `test/test_drift_sim.py`.

## Notes / out of scope (v1)

On-demand only (no scheduled compute), unauthenticated (trusted-network
posture, like the inference API), single-feature univariate drift. No
`mediwatch.sh` command surfacing yet: `docker compose up -d` starts it.
