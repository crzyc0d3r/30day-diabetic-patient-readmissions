"""Unit tests for helpers/cicd_trigger.py — the remote CI/CD build+deploy
trigger used by the retrain_on_drift DAG to orchestrate deploys remotely instead
of running kubectl inline.

The suite is hermetic: it never makes a network call. Every request is built
against an in-test env dict, and trigger_remote_deploy is exercised with a fake
``poster`` that records the call and returns a canned (status, body). This keeps
the tests fast and lets us assert on the exact URL, headers, and body each
provider sends.

Per project conventions this file avoids em dashes and semicolons and uses the
spelling "program".
"""
from __future__ import annotations

import base64
import json

import pytest

from helpers.cicd_trigger import (
    PROVIDERS,
    RemoteTriggerError,
    _azure_request,
    _cloudbuild_request,
    _jenkins_request,
    select_provider,
    trigger_remote_deploy,
)

JENKINS_ENV = {
    "JENKINS_URL": "https://ci.example.com/",
    "JENKINS_JOB": "medi-watch",
    "JENKINS_USER": "airflow",
    "JENKINS_API_TOKEN": "tok123",
}
AZURE_ENV = {
    "AZURE_DEVOPS_ORG": "acme",
    "AZURE_DEVOPS_PROJECT": "medi-watch",
    "AZURE_PIPELINE_ID": "42",
    "AZURE_DEVOPS_PAT": "pat456",
}
CLOUDBUILD_ENV = {
    "GCP_PROJECT": "proj-1",
    "GCB_TRIGGER": "medi-watch-trigger",
    "GOOGLE_OAUTH_TOKEN": "ya29.token",
}


# --------------------------------------------------------------------------- #
# select_provider
# --------------------------------------------------------------------------- #
def test_select_provider_explicit_wins():
    for p in PROVIDERS:
        assert select_provider({"CICD_PROVIDER": p}) == p
    assert select_provider({"CICD_PROVIDER": "JENKINS"}) == "jenkins"  # case-insensitive


def test_select_provider_explicit_none_opts_out():
    for word in ("none", "off", "disabled"):
        assert select_provider({"CICD_PROVIDER": word}) is None


def test_select_provider_unknown_raises():
    with pytest.raises(RemoteTriggerError):
        select_provider({"CICD_PROVIDER": "travis"})


def test_select_provider_autodetects_by_primary_credential():
    assert select_provider({"JENKINS_URL": "x"}) == "jenkins"
    assert select_provider({"AZURE_DEVOPS_ORG": "x"}) == "azure"
    assert select_provider({"GCB_TRIGGER": "x"}) == "cloudbuild"


def test_select_provider_none_when_unconfigured():
    assert select_provider({}) is None


# --------------------------------------------------------------------------- #
# per-provider request building
# --------------------------------------------------------------------------- #
def test_jenkins_request_shape_and_auth():
    url, headers, data = _jenkins_request(JENKINS_ENV, "promote", True)
    assert url.startswith("https://ci.example.com/job/medi-watch/buildWithParameters?")
    assert "REASON=promote" in url and "SKIP_SMOKE=true" in url
    assert data == b""
    expected = base64.b64encode(b"airflow:tok123").decode()
    assert headers["Authorization"] == f"Basic {expected}"


def test_jenkins_request_includes_build_token_when_set():
    url, _, _ = _jenkins_request({**JENKINS_ENV, "JENKINS_BUILD_TOKEN": "secret"}, "r", False)
    assert "token=secret" in url and "SKIP_SMOKE=false" in url


def test_jenkins_request_missing_env_lists_keys():
    with pytest.raises(RemoteTriggerError) as exc:
        _jenkins_request({"JENKINS_URL": "x"}, "r", True)
    assert "JENKINS_JOB" in str(exc.value)


def test_azure_request_shape_body_and_auth():
    url, headers, data = _azure_request(AZURE_ENV, "bootstrap", False)
    assert url == (
        "https://dev.azure.com/acme/medi-watch/_apis/pipelines/42/runs"
        "?api-version=7.1-preview.1"
    )
    payload = json.loads(data)
    assert payload["templateParameters"] == {"reason": "bootstrap", "skipSmoke": False}
    assert payload["resources"]["repositories"]["self"]["refName"] == "refs/heads/main"
    expected = base64.b64encode(b":pat456").decode()
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["Content-Type"] == "application/json"


def test_azure_request_honours_custom_branch():
    _, _, data = _azure_request({**AZURE_ENV, "AZURE_PIPELINE_BRANCH": "refs/heads/release"}, "r", True)
    assert json.loads(data)["resources"]["repositories"]["self"]["refName"] == "refs/heads/release"


def test_cloudbuild_request_shape_body_and_auth():
    url, headers, data = _cloudbuild_request(CLOUDBUILD_ENV, "promote", True)
    assert url == "https://cloudbuild.googleapis.com/v1/projects/proj-1/triggers/medi-watch-trigger:run"
    payload = json.loads(data)
    assert payload["branchName"] == "main"
    assert payload["substitutions"] == {"_REASON": "promote", "_SKIP_SMOKE": "true"}
    assert headers["Authorization"] == "Bearer ya29.token"


def test_cloudbuild_request_missing_env_raises():
    with pytest.raises(RemoteTriggerError):
        _cloudbuild_request({"GCP_PROJECT": "p"}, "r", True)


# --------------------------------------------------------------------------- #
# trigger_remote_deploy dispatch
# --------------------------------------------------------------------------- #
class _Recorder:
    """A fake poster capturing the last call and returning a canned response."""

    def __init__(self, status=201, body="{}"):
        self.status, self.body = status, body
        self.calls = []

    def __call__(self, url, headers, data):
        self.calls.append((url, headers, data))
        return self.status, self.body


def test_trigger_success_returns_result_and_posts_once():
    poster = _Recorder(status=201, body='{"id": 7}')
    result = trigger_remote_deploy(env=JENKINS_ENV, reason="promote", poster=poster)
    assert result["provider"] == "jenkins"
    assert result["ok"] is True and result["status"] == 201
    assert result["reason"] == "promote"
    assert len(poster.calls) == 1
    assert "buildWithParameters" in poster.calls[0][0]


def test_trigger_non_2xx_raises():
    poster = _Recorder(status=403, body="forbidden")
    with pytest.raises(RemoteTriggerError) as exc:
        trigger_remote_deploy(env=AZURE_ENV, poster=poster)
    assert "403" in str(exc.value)


def test_trigger_no_provider_configured_raises():
    with pytest.raises(RemoteTriggerError):
        trigger_remote_deploy(env={}, poster=_Recorder())


def test_trigger_explicit_provider_overrides_autodetect():
    # env carries both jenkins + cloudbuild creds; explicit provider wins.
    poster = _Recorder()
    merged = {**JENKINS_ENV, **CLOUDBUILD_ENV}
    result = trigger_remote_deploy(provider="cloudbuild", env=merged, poster=poster)
    assert result["provider"] == "cloudbuild"
    assert poster.calls[0][0].endswith(":run")
