"""Shared helpers for the medi-watch pipeline and services.

This package is the single source of truth for cleaning rules, model
factories, lab Postgres publishing, and MLflow logging used by every
notebook (01–10), the inference API, the Airflow DAGs, and the
GenAI testing lab. Importing from anywhere else inside the repo means
two implementations are about to drift. Add a function here instead.
"""
