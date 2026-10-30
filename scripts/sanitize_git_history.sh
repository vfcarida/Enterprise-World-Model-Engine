#!/usr/bin/env bash
set -euo pipefail

# Git history sanitization script for Enterprise World Model Engine.
# WARNING: This permanently rewrites Git history. Run ONLY after creating a backup branch.

echo "=== Enterprise World Model Engine: Git History Sanitization ==="
echo "This script will PERMANENTLY remove:"
echo "  - All instructional scaffolding files"
echo "  - All workbench and scratch artifacts"
echo "  - Any non-author attribution trailers in commit messages"
echo ""
read -r -p "Have you created a backup branch? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborting. Create backup branch first: git branch backup-pre-sanitization"
    exit 1
fi

if ! command -v git-filter-repo &> /dev/null; then
    echo "Installing git-filter-repo..."
    pip install git-filter-repo
fi

echo "Step 1: Removing instructional scaffolding files..."
git filter-repo --path-glob '**/PROMPT-*.md' --invert-paths --force

echo "Step 2: Removing workbench artifacts..."
git filter-repo --path-glob '**/workbench/**' --invert-paths --force

echo "Step 3: Redacting instructional scaffolding markers..."
git filter-repo --replace-text <(cat <<'EOF'
regex:You are acting as.*==>[REDACTED]
regex:System prompt:.*==>[REDACTED]
regex:Chain-of-thought:.*==>[REDACTED]
regex:LLM instruction:.*==>[REDACTED]
EOF
) --force

echo "Step 4: Sanitizing commit messages..."
git filter-repo --message-callback '
import re
msg = message.decode("utf-8", errors="replace")
msg = re.sub(r"(Generated|Implemented|Created) (with|via|using) (Claude|GPT-4|Copilot|AI assistant)", r"\1", msg, flags=re.IGNORECASE)
msg = re.sub(r"Co-Authored-By:.*\n?", "", msg, flags=re.IGNORECASE)
return msg.encode("utf-8")
' --force

echo "Git history sanitization complete."
echo "Next steps:"
echo "  1. Verify: git log --all --oneline | head -20"
echo "  2. Inspect: git show HEAD"
echo "  3. Force-push to remote: git push origin --force --all"
