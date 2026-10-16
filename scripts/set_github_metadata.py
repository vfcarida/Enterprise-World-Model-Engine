"""Automated script to sync GitHub repository metadata via the REST API.

Synchronizes:
1. One-line mechanism-first description
2. Homepage URL (GitHub Pages docs site)
3. Topics (deterministic-simulation, provenance, causal, digital-twin, scenario-planning,
   decision-intelligence, what-if, monte-carlo, simulation, counterfactual)
4. GitHub Discussions enablement
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

REPO_OWNER = "vfcarida"
REPO_NAME = "Enterprise-World-Model-Engine"

TARGET_DESCRIPTION = "Deterministic, auditable what-if simulation kernel for socio-technical systems — no GPU/LLM required"
TARGET_HOMEPAGE = "https://vfcarida.github.io/Enterprise-World-Model-Engine/"
TARGET_TOPICS = [
    "deterministic-simulation",
    "provenance",
    "causal",
    "digital-twin",
    "scenario-planning",
    "decision-intelligence",
    "what-if",
    "monte-carlo",
    "simulation",
    "counterfactual",
]


def sync_github_metadata(token: str) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ewm-engine-metadata-sync",
    }

    # 1. Update Repository Details (description, homepage, has_discussions)
    repo_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    repo_data = {
        "description": TARGET_DESCRIPTION,
        "homepage": TARGET_HOMEPAGE,
        "has_discussions": True,
    }

    print(f"Updating repository properties at {repo_url}...")
    req = urllib.request.Request(
        repo_url,
        data=json.dumps(repo_data).encode("utf-8"),
        headers=headers,
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))
            print(f"SUCCESS: Description set to: {res_body.get('description')}")
            print(f"SUCCESS: Homepage set to: {res_body.get('homepage')}")
            print(f"SUCCESS: Discussions enabled: {res_body.get('has_discussions')}")
    except urllib.error.HTTPError as err:
        print(f"HTTPError updating repo details: {err.code} {err.reason}")
        print(err.read().decode("utf-8"))
        if err.code == 403 or err.code == 401:
            print("Note: Token requires repository administration / write permissions.")

    # 2. Update Repository Topics
    topics_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/topics"
    topics_data = {"names": TARGET_TOPICS}
    print(f"\nUpdating repository topics at {topics_url}...")
    req_topics = urllib.request.Request(
        topics_url,
        data=json.dumps(topics_data).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req_topics) as resp:
            topics_res = json.loads(resp.read().decode("utf-8"))
            print(f"SUCCESS: Topics set to: {topics_res.get('names')}")
    except urllib.error.HTTPError as err:
        print(f"HTTPError updating topics: {err.code} {err.reason}")
        print(err.read().decode("utf-8"))


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("INFO: GITHUB_TOKEN / GH_TOKEN environment variable is not set.")
        print("Target repository settings are:")
        print(f"  Description: {TARGET_DESCRIPTION}")
        print(f"  Homepage:    {TARGET_HOMEPAGE}")
        print(f"  Topics:      {', '.join(TARGET_TOPICS)}")
        print("  Discussions: Enabled")
        print(
            "To apply live via GitHub API, run: GITHUB_TOKEN=... python scripts/set_github_metadata.py"
        )
        return 0

    sync_github_metadata(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
