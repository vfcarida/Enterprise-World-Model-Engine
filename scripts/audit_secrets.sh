#!/usr/bin/env bash
set -euo pipefail

echo "=== Enterprise World Model Engine: Credential & Secret Audit ==="

if command -v gitleaks &> /dev/null; then
    echo "Scanning Git history with gitleaks..."
    gitleaks detect --source . --report-path gitleaks-report.json --verbose
    if [ -f gitleaks-report.json ] && [ -s gitleaks-report.json ]; then
        echo "ERROR: Secrets detected! Review gitleaks-report.json"
        cat gitleaks-report.json
        exit 1
    else
        echo "OK: No secrets detected in Git history."
    fi
else
    echo "Note: gitleaks binary not found on PATH; running repository secret test suite..."
    python -m pytest tests/security/test_no_credentials.py -q
fi

echo "Scanning working tree for hardcoded credentials..."
python -m pytest tests/security/test_no_credentials.py -q
echo "OK: Credential and secret audit complete."
