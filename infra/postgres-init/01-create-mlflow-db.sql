-- First-init seed for the MLflow backend store.
--
-- Runs once when the Postgres container initialises an empty data volume.
-- Airflow owns the default DB declared in POSTGRES_DB (env at container
-- start). This script adds a second logical DB named `mlflow` so tracking
-- metadata and experiment runs are isolated from Airflow's metadata tables,
-- which is convenient for backups and for keeping `mlflow gc` from touching DAG
-- state. The two databases share the same POSTGRES_USER credentials
-- injected via infra/.env (see infra/.env.example for the full set).
CREATE DATABASE mlflow;
