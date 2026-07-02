"""Demonstrate the Airflow -> CI remote-trigger handoff end to end.

Spins up a local mock CI server, then fires `helpers.cicd_trigger` (the same code
path `retrain_on_drift_dag.trigger_remote_deploy` runs) at it, capturing the exact
HTTP request the orchestration sends and the CI system's response. This is
reproducible evidence that the orchestration layer remotely triggers a build+deploy
job rather than running kubectl inline.

Run:  python infra/ci-cd/demo_remote_trigger.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.cicd_trigger import (  # noqa: E402
    _redact,
    build_request,
    trigger_remote_deploy,
)

_captured: dict = {}


class _MockCI(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence default logging
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _captured.update(
            method=self.command,
            path=self.path,
            authorization=self.headers.get("Authorization", ""),
            body=self.rfile.read(n).decode() if n else "",
        )
        self.send_response(201)  # Jenkins returns 201 Created with a queue Location
        self.send_header("Location", "http://mock-ci/queue/item/42/")
        self.end_headers()
        self.wfile.write(b'{"queued": true, "build_number": 42}')


def main() -> int:
    print("=== 1) dry-run: the request the orchestration WOULD send, per provider ===")
    samples = [
        ("jenkins", {"JENKINS_URL": "https://ci.example.com", "JENKINS_JOB": "medi-watch",
                     "JENKINS_USER": "airflow-deployer", "JENKINS_API_TOKEN": "tok",
                     "JENKINS_BUILD_TOKEN": "btok"}),
        ("azure", {"AZURE_DEVOPS_ORG": "acme", "AZURE_DEVOPS_PROJECT": "medi-watch",
                   "AZURE_PIPELINE_ID": "42", "AZURE_DEVOPS_PAT": "pat"}),
        ("cloudbuild", {"GCP_PROJECT": "proj-1", "GCB_TRIGGER": "medi-watch-trigger",
                        "GOOGLE_OAUTH_TOKEN": "ya29.tok"}),
    ]
    reason = "retrain_on_drift: promote (scenario=mixed_severe)"
    for provider, env in samples:
        url, headers, data = build_request(provider, env, reason, skip_smoke=True)
        print(f"\n[{provider}] POST {url}")
        print(f"   headers: {_redact(headers)}")
        print(f"   body:    {data.decode('utf-8', 'replace') or '(empty)'}")

    print("\n=== 2) LIVE fire against a local mock CI server (Jenkins shape) ===")
    server = HTTPServer(("127.0.0.1", 0), _MockCI)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        "JENKINS_URL": f"http://127.0.0.1:{port}", "JENKINS_JOB": "medi-watch",
        "JENKINS_USER": "airflow-deployer", "JENKINS_API_TOKEN": "demo-token",
        "JENKINS_BUILD_TOKEN": "demo-build-token",
    }
    try:
        result = trigger_remote_deploy(provider="jenkins", env=env, reason=reason, skip_smoke=True)
    finally:
        server.shutdown()

    print("\ntrigger_remote_deploy() returned:")
    print(json.dumps(result, indent=2))
    print("\nwhat the mock CI server received:")
    print(json.dumps(_captured, indent=2))

    assert result["ok"] and result["status"] == 201, "trigger did not get a 2xx"
    assert "buildWithParameters" in _captured["path"], "did not hit the build endpoint"
    assert "REASON=" in _captured["path"], "REASON parameter not forwarded"
    print("\nOK: orchestration -> HTTP POST -> CI build job accepted (HTTP 201). Handoff verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
