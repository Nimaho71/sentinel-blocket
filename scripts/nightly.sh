#!/bin/bash
#
# The nightly run: sweep, evaluate, publish, and push the public snapshot.
#
# Lives in the repository rather than inside the launchd plist so that the
# logic is reviewable and version-controlled. The plist should contain nothing
# but a schedule and a path to this file.
#
#   Manual (do this first, and watch it):
#       MAX_DURATION_MIN=120 scripts/nightly.sh
#
#   Scheduled:
#       ~/Library/LaunchAgents/se.sentinel.nightly.plist
#
# Every phase gates the next. A failed publish must never reach the push, and a
# run that could not verify its own output must not leave anything behind.

set -euo pipefail

# launchd does not run a login shell, so nothing here is inherited: not PATH,
# not the working directory, not anything from .zshrc. Everything the run needs
# is stated explicitly below. This is the single largest source of "works in my
# terminal, silently broken at 03:00".
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# `ai/gemini_client.py` resolves QUOTA_LEDGER_PATH relative to the working
# directory. launchd starts jobs in `/`, so without an absolute path here the
# job would create a fresh, empty quota ledger somewhere else every single
# night and the quota accounting would quietly stop meaning anything.
export QUOTA_LEDGER_PATH="$REPO/quota_ledger.db"

PYTHON="$REPO/venv/bin/python"
SNAPSHOT="web/public/deals.json"
LOG_DIR="$HOME/Library/Logs/sentinel"
LOCK="$REPO/.nightly.lock"

# Wall-clock ceiling for the sweep. 60 is a safety net rather than a schedule:
# a steady-state run finishes well inside it, and the cap only bites on the
# first deep sweep or when catching up after an outage. Override for the first
# manual run, which has thousands of unseen ads to fetch and will take longer.
MAX_DURATION_MIN="${MAX_DURATION_MIN:-60}"

mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "FAILED: $*"; exit 1; }

# --------------------------------------------------------------------- locking

# `mkdir` is atomic on every filesystem this will ever run on, which `[ -f ]`
# followed by `touch` is not. Two pipelines writing one SQLite file is how you
# corrupt an index, and an overrunning run must not have tomorrow's start on
# top of it.
if ! mkdir "$LOCK" 2>/dev/null; then
    # A held lock is not proof of a live run. The trap below never fires on a
    # power cut or a SIGKILL, so the directory outlives the process — and a
    # lock that only a human can clear turns one crash into a pipeline that is
    # silently dead every night from then on. Exactly the failure this project
    # keeps meeting: everything reports fine, the data is just old.
    holder=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
        die "run $holder still holds $LOCK (started $(date -r "$LOCK" '+%H:%M:%S'))"
    fi
    log "WARNING: reclaiming $LOCK from dead process ${holder:-unknown} —"
    log "         the previous run was killed rather than finishing."
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || die "could not reclaim $LOCK"
fi
echo "$$" > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT

log "=== nightly run starting (max-duration ${MAX_DURATION_MIN}m) ==="

# ----------------------------------------------------------------- environment

[ -f .env ] || die ".env not found. The pipeline needs GEMINI_API_KEY."
set -a
# shellcheck disable=SC1091
. ./.env
set +a
[ -n "${GEMINI_API_KEY:-}" ] || die "GEMINI_API_KEY is empty in .env"
[ -x "$PYTHON" ] || die "no interpreter at $PYTHON — is the venv built?"

# ------------------------------------------------------------------- preflight

# Checked before the scrape, not after. Discovering that the push is impossible
# is worth knowing before spending twenty minutes of Blocket's bandwidth on a
# snapshot that cannot leave the machine.
git fetch --quiet origin main || die "cannot reach origin — check network and credentials"

if ! git diff --cached --quiet; then
    die "there are already-staged changes in the working tree. Refusing to run:
     this job commits $SNAPSHOT and nothing else, and an unattended commit
     must not sweep up whatever was left staged."
fi

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse origin/main)"
PUSHABLE=1
if [ "$LOCAL" != "$REMOTE" ]; then
    # Deliberately does not rebase. Resolving a divergence unattended, at 03:00,
    # against a repository whose only output is a published artifact, is a much
    # worse failure than skipping one night's push. The snapshot is still built
    # and still on disk; the next manual push carries it.
    PUSHABLE=0
    log "WARNING: HEAD and origin/main have diverged. The run will proceed and"
    log "         write the snapshot, but will NOT commit or push. Reconcile by hand."
fi

# ------------------------------------------------------------------- the sweep

log "--- pipeline ---"
"$PYTHON" -m pipeline.run --max-duration "$MAX_DURATION_MIN" \
    || die "pipeline exited non-zero — nothing published, nothing pushed"

# `publish` refuses to write anything it could not verify, so a file that
# exists has already passed its redaction check. Re-running the independent
# checker costs nothing and is the one guard that shares no code with the
# builder, which is the entire reason it is trusted.
log "--- verifying snapshot ---"
"$PYTHON" -m pipeline.redact "$SNAPSHOT" || die "snapshot failed its redaction check"

# ------------------------------------------------------------------ publishing

if [ "$PUSHABLE" -eq 0 ]; then
    log "skipping commit and push (diverged from origin/main)"
    log "=== nightly run finished: snapshot written, not published ==="
    exit 0
fi

if git diff --quiet -- "$SNAPSHOT"; then
    log "snapshot unchanged — nothing to publish"
    log "=== nightly run finished ==="
    exit 0
fi

log "--- publishing ---"
# One explicit path. Never `git add -A`: at 03:00 the working tree may hold
# half-finished edits, and this job has no business deciding those are ready.
git add "$SNAPSHOT"
git commit --quiet -m "data: nightly snapshot $(date '+%Y-%m-%d')" \
    || die "commit rejected — the pre-commit hook blocks credentials and unredacted data"
git push --quiet origin main || die "push failed — snapshot is committed locally but not published"

log "=== nightly run finished: published $(git rev-parse --short HEAD) ==="
