#!/bin/bash
# Full pipeline: curated frames -> plain text -> hand-written sub-steps -> extra approaches -> real-code traces -> index
set -e
cd "$(dirname "$0")"
node old7.js
node build.js | tail -1
node apply.js | tail -1
node subs.js
node alts_apply.js
node dump.js
python3 trace.py
node inject.js
node index.js
