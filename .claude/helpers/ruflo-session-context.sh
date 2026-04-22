#!/usr/bin/env bash
# SessionStart hook: surface current ruflo state into Claude's context.
# Enforces the standing directive "Always use Ruflo!" — guarantees Claude sees
# ruflo swarm status + recent memory entries at the start of every session.
#
# Silent-safe: if ruflo isn't on PATH the hook still emits a valid JSON
# additionalContext (just with a soft reminder to install/init ruflo) and never
# fails the session.

set -u

readarray_safe() {
  local _out
  _out="$("$@" 2>&1)" || _out="(command failed: $*)"
  printf '%s\n' "$_out"
}

if ! command -v ruflo >/dev/null 2>&1; then
  SWARM="(ruflo binary not on PATH — install with 'npm i -g ruflo')"
  MEMORY="(ruflo memory unavailable)"
else
  SWARM="$(readarray_safe ruflo swarm status)"
  MEMORY="$(readarray_safe ruflo memory list --limit 10)"
fi

python - <<PY 2>/dev/null || true
import json, os

ctx = (
    '[ruflo session start — per user directive "Always use Ruflo!"]\n'
    '\n=== ruflo swarm status ===\n'
    + os.environ.get('SWARM', '') +
    '\n\n=== ruflo memory (latest 10 entries) ===\n'
    + os.environ.get('MEMORY', '') +
    '\n\nReminder: '
    'search ruflo memory before non-trivial decisions '
    "(ruflo memory search -q \"<topic>\"); "
    "write significant decisions back "
    "(ruflo memory store -k <key> --value <text>); "
    "at session close, persist a session/close/<date> summary entry."
)

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": ctx,
    }
}))
PY
