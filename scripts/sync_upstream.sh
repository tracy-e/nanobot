#!/bin/bash
# Sync upstream (origin/main) into current branch with validation.
#
# Usage:
#   ./scripts/sync_upstream.sh          # rebase onto origin/main
#   ./scripts/sync_upstream.sh merge    # merge origin/main instead
#
# Flow:
#   1. Stash uncommitted changes
#   2. Fetch origin
#   3. Rebase (or merge) onto origin/main
#   4. Run validate_sync.py
#   5. If validation fails → abort and restore
#   6. If validation passes → pop stash, done

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
RESET='\033[0m'

STRATEGY="${1:-rebase}"
BRANCH=$(git branch --show-current)
PYTHON="${PYTHON:-python3}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Syncing upstream → ${BRANCH}"
echo "  Strategy: ${STRATEGY}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Stash
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}Stashing uncommitted changes...${RESET}"
    git stash push -m "sync_upstream auto-stash"
    STASHED=true
fi

# 2. Fetch
echo "Fetching origin..."
git fetch origin

# 3. Record current HEAD for rollback
ORIG_HEAD=$(git rev-parse HEAD)

# 4. Rebase or merge
if [ "$STRATEGY" = "merge" ]; then
    echo "Merging origin/main..."
    if ! git merge origin/main --no-edit; then
        echo -e "${RED}Merge conflicts detected. Resolve and re-run.${RESET}"
        exit 1
    fi
else
    echo "Rebasing onto origin/main..."
    if ! git rebase origin/main; then
        echo -e "${RED}Rebase conflicts detected.${RESET}"
        echo "Resolve conflicts, then run: python3 scripts/validate_sync.py"
        exit 1
    fi
fi

# 5. Validate
echo ""
if $PYTHON scripts/validate_sync.py; then
    echo ""
    echo -e "${GREEN}Sync successful!${RESET}"
else
    echo ""
    echo -e "${RED}Validation failed! Rolling back...${RESET}"
    if [ "$STRATEGY" = "merge" ]; then
        git reset --hard "$ORIG_HEAD"
    else
        git rebase --abort 2>/dev/null || git reset --hard "$ORIG_HEAD"
    fi
    echo "Restored to $(git rev-parse --short HEAD)"
    if $STASHED; then
        git stash pop
    fi
    exit 1
fi

# 6. Restore stash
if $STASHED; then
    echo "Restoring stashed changes..."
    git stash pop
fi

echo -e "${GREEN}Done. Run 'launchctl stop com.nanobot.gateway && sleep 1 && launchctl start com.nanobot.gateway' to restart.${RESET}"
