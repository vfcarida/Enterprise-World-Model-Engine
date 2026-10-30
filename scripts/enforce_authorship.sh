#!/usr/bin/env bash
set -euo pipefail

echo "=== Enterprise World Model Engine: Authorship Attribution Enforcement ==="

git config user.name "Vinicius Caridá"
git config user.email "vfcarida@gmail.com"

echo "Verifying commit authorship..."
non_author_commits=$(git log --all --format="%an <%ae>" | grep -v -E "Vinicius Caridá|dependabot\[bot\]|github-actions\[bot\]" || true)

if [ -n "$non_author_commits" ]; then
    echo "WARNING: Found commits by unexpected authors:"
    echo "$non_author_commits"
    echo ""
    echo "To fix: git filter-repo --mailmap <(echo 'Vinicius Caridá <vfcarida@gmail.com> <old-email>')"
    exit 1
else
    echo "OK: All human commits authored by Vinicius Caridá."
fi

echo "Scanning for Co-Authored-By trailers..."
co_authored=$(git log --all --grep="Co-Authored-By" --oneline || true)

if [ -n "$co_authored" ]; then
    echo "ERROR: Found Co-Authored-By trailers. Re-run git-filter-repo message-callback."
    exit 1
else
    echo "OK: No Co-Authored-By trailers found."
fi

echo "Authorship enforcement complete."
