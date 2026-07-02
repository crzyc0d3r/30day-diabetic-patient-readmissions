"""Remote-trigger the medi-watch CI/CD build+deploy job from the orchestration
layer.

The Airflow ``retrain_on_drift`` DAG promotes a new ``@champion`` and then needs
the inference API rebuilt + redeployed. Doing that with a direct ``kubectl
rollout restart`` from inside the worker couples orchestration to cluster access
and skips the CI pipeline entirely. This module lets the DAG instead **remotely
trigger** the existing build+deploy job (``infra/ci-cd/{Jenkinsfile,
azure-pipelines.yml,cloudbuild.yaml}``) on whichever CI system is configured, so
"build and deployment jobs" are orchestrated remotely rather than run inline.

All three providers accept the same intent — a ``REASON`` string and a
``SKIP_SMOKE`` flag — mirroring the parameters those pipelines already declare.

Design notes:
  * Stdlib only (``urllib``), so it runs unmodified in the Airflow image with no
    extra dependency. The single network call goes through ``_http_post``, which
    tests replace with a fake ``poster`` so the suite is hermetic.
  * Provider is chosen by ``CICD_PROVIDER`` (explicit) or auto-detected from
    whichever provider's primary credential is present. With none configured,
    ``select_provider`` returns ``None`` and the DAG falls back to the direct
    ``kubectl`` path (local/dev).
  * Every builder validates its required env up front and raises
    ``RemoteTriggerError`` listing what is missing, rather than firing a
    half-formed request.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

PROVIDERS = ("jenkins", "azure", "cloudbuild")

# Env keys whose presence auto-selects a provider when CICD_PROVIDER is unset.
_PRIMARY_CRED = {
    "jenkins": "JENKINS_URL",
    "azure": "AZURE_DEVOPS_ORG",
    "cloudbuild": "GCB_TRIGGER",
}


class RemoteTriggerError(RuntimeError):
    """Raised on missing configuration or a non-2xx response from the CI system."""


def _require(env: Mapping[str, str], keys: list[str]) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise RemoteTriggerError(
            "missing required environment for the remote trigger: " + ", ".join(missing)
        )


def select_provider(env: Mapping[str, str] | None = None) -> str | None:
    """Resolve which CI/CD provider to trigger.

    ``CICD_PROVIDER`` wins when set (``none``/``off``/``disabled`` explicitly
    opts out). Otherwise auto-detect by the presence of each provider's primary
    credential. Returns ``None`` when nothing is configured.
    """
    env = os.environ if env is None else env
    explicit = (env.get("CICD_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit in ("none", "off", "disabled", ""):
            return None
        if explicit not in PROVIDERS:
            raise RemoteTriggerError(
                f"unknown CICD_PROVIDER '{explicit}' (expected one of "
                f"{', '.join(PROVIDERS)} or 'none')"
            )
        return explicit
    for provider in PROVIDERS:
        if env.get(_PRIMARY_CRED[provider]):
            return provider
    return None


def _jenkins_request(env, reason, skip_smoke):
    """Jenkins remote build: POST {JENKINS_URL}/job/{JENKINS_JOB}/buildWithParameters.

    Authenticates with an API token (Basic auth); an optional per-job build
    ``token`` is added as a query param when ``JENKINS_BUILD_TOKEN`` is set.
    """
    _require(env, ["JENKINS_URL", "JENKINS_JOB", "JENKINS_USER", "JENKINS_API_TOKEN"])
    base = env["JENKINS_URL"].rstrip("/")
    params = {"REASON": reason, "SKIP_SMOKE": "true" if skip_smoke else "false"}
    if env.get("JENKINS_BUILD_TOKEN"):
        params["token"] = env["JENKINS_BUILD_TOKEN"]
    url = f"{base}/job/{env['JENKINS_JOB']}/buildWithParameters?" + urllib.parse.urlencode(params)
    cred = base64.b64encode(
        f"{env['JENKINS_USER']}:{env['JENKINS_API_TOKEN']}".encode()
    ).decode()
    return url, {"Authorization": f"Basic {cred}"}, b""


def _azure_request(env, reason, skip_smoke):
    """Azure Pipelines run: POST .../pipelines/{id}/runs with a PAT (Basic auth)."""
    _require(env, ["AZURE_DEVOPS_ORG", "AZURE_DEVOPS_PROJECT", "AZURE_PIPELINE_ID", "AZURE_DEVOPS_PAT"])
    org = env["AZURE_DEVOPS_ORG"]
    project = env["AZURE_DEVOPS_PROJECT"]
    pipeline_id = env["AZURE_PIPELINE_ID"]
    branch = env.get("AZURE_PIPELINE_BRANCH", "refs/heads/main")
    url = (
        f"https://dev.azure.com/{org}/{project}/_apis/pipelines/{pipeline_id}/runs"
        "?api-version=7.1-preview.1"
    )
    body = json.dumps({
        "resources": {"repositories": {"self": {"refName": branch}}},
        "templateParameters": {"reason": reason, "skipSmoke": skip_smoke},
    }).encode()
    # PAT is sent as the password with an empty username.
    cred = base64.b64encode(f":{env['AZURE_DEVOPS_PAT']}".encode()).decode()
    return url, {"Authorization": f"Basic {cred}", "Content-Type": "application/json"}, body


def _cloudbuild_request(env, reason, skip_smoke):
    """Google Cloud Build: POST .../triggers/{trigger}:run with an OAuth token."""
    _require(env, ["GCP_PROJECT", "GCB_TRIGGER", "GOOGLE_OAUTH_TOKEN"])
    project = env["GCP_PROJECT"]
    trigger = env["GCB_TRIGGER"]
    branch = env.get("GCB_BRANCH", "main")
    url = f"https://cloudbuild.googleapis.com/v1/projects/{project}/triggers/{trigger}:run"
    body = json.dumps({
        "branchName": branch,
        "substitutions": {"_REASON": reason, "_SKIP_SMOKE": "true" if skip_smoke else "false"},
    }).encode()
    headers = {
        "Authorization": f"Bearer {env['GOOGLE_OAUTH_TOKEN']}",
        "Content-Type": "application/json",
    }
    return url, headers, body


_BUILDERS = {
    "jenkins": _jenkins_request,
    "azure": _azure_request,
    "cloudbuild": _cloudbuild_request,
}


def _http_post(url: str, headers: dict[str, str], data: bytes) -> tuple[int, str]:
    """POST via stdlib urllib. Returns ``(status, body)``; an HTTP error status
    is returned (not raised) so the caller can surface the CI system's message."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def trigger_remote_deploy(
    *,
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
    reason: str = "champion-promotion",
    skip_smoke: bool = True,
    poster: Callable[[str, dict, bytes], tuple[int, str]] = _http_post,
) -> dict:
    """Trigger the configured CI/CD build+deploy job and return a result dict.

    Raises ``RemoteTriggerError`` when no provider is configured, the provider's
    required env is missing, or the CI system returns a non-2xx status.
    """
    env = os.environ if env is None else env
    provider = provider or select_provider(env)
    if provider is None:
        raise RemoteTriggerError("no CI/CD provider configured (set CICD_PROVIDER)")
    if provider not in _BUILDERS:
        raise RemoteTriggerError(f"unknown provider '{provider}'")

    url, headers, data = _BUILDERS[provider](env, reason, skip_smoke)
    status, body = poster(url, headers, data)
    ok = 200 <= status < 300
    result = {
        "provider": provider,
        "status": status,
        "ok": ok,
        "url": url,
        "reason": reason,
        "skip_smoke": skip_smoke,
        "response": body[:500],
    }
    if not ok:
        raise RemoteTriggerError(
            f"{provider} trigger failed: HTTP {status}: {body[:300]}"
        )
    return result


def build_request(provider: str, env: Mapping[str, str], reason: str,
                  skip_smoke: bool) -> tuple[str, dict, bytes]:
    """Return the exact (url, headers, body) that ``trigger_remote_deploy`` would
    POST, without sending it. Used by the ``--dry-run`` CLI to show the request."""
    if provider not in _BUILDERS:
        raise RemoteTriggerError(f"unknown provider '{provider}'")
    return _BUILDERS[provider](env, reason, skip_smoke)


def _redact(headers: dict) -> dict:
    """Mask credential headers so a dry-run trace is safe to print/commit."""
    out = dict(headers)
    if "Authorization" in out:
        scheme = out["Authorization"].split(" ", 1)[0]
        out["Authorization"] = f"{scheme} <redacted>"
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: fire (or dry-run) the configured CI/CD build+deploy job by hand.

    Examples:
        python -m helpers.cicd_trigger --dry-run
        python -m helpers.cicd_trigger --provider jenkins --reason "manual smoke"
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m helpers.cicd_trigger",
        description="Remotely trigger the medi-watch CI/CD build+deploy job "
                    "(Jenkins / Azure Pipelines / Google Cloud Build).")
    parser.add_argument("--provider", choices=PROVIDERS,
                        help="CI provider (default: CICD_PROVIDER env or auto-detect).")
    parser.add_argument("--reason", default="manual cicd_trigger CLI",
                        help="Free-text reason recorded in the CI run.")
    parser.add_argument("--skip-smoke", dest="skip_smoke", action="store_true", default=True,
                        help="Tell the pipeline to skip the e2e smoke stage (default).")
    parser.add_argument("--no-skip-smoke", dest="skip_smoke", action="store_false",
                        help="Run the pipeline's full e2e smoke stage.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the request that would be sent, without sending it.")
    args = parser.parse_args(argv)

    provider = args.provider or select_provider(os.environ)
    if provider is None:
        print("No CI/CD provider configured. Set CICD_PROVIDER (jenkins|azure|cloudbuild) "
              "and that provider's credentials, or pass --provider.")
        return 2

    if args.dry_run:
        url, headers, data = build_request(provider, os.environ, args.reason, args.skip_smoke)
        print(f"provider : {provider}")
        print(f"method   : POST")
        print(f"url      : {url}")
        print(f"headers  : {_redact(headers)}")
        print(f"body     : {data.decode('utf-8', 'replace') or '(empty)'}")
        print("(dry-run: nothing sent)")
        return 0

    try:
        result = trigger_remote_deploy(provider=provider, reason=args.reason,
                                       skip_smoke=args.skip_smoke)
    except RemoteTriggerError as exc:
        print(f"trigger failed: {exc}")
        return 1
    print(f"triggered {result['provider']} build+deploy: HTTP {result['status']} "
          f"(reason={result['reason']!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
