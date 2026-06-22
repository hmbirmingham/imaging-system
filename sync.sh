#!/bin/bash
# sync.sh — push local vscode folder to GitHub
# Usage: ./sync.sh "optional commit message"

MSG=${1:-"Update"}

git add -A
git commit -m "$MSG" 2>/dev/null || echo "(nothing new to commit)"
git push origin main --force-with-lease
echo "Done."
