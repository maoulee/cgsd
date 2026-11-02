#!/usr/bin/env bash
# Bootstrap a local Freebase Virtuoso SPARQL endpoint for the repair step.
#
# The dump we used is the processed Freebase Virtuoso DB distributed by the
# ChatKBQA project (LHRLAB/ChatKBQA), which packages the dki-lab/Freebase-Setup
# service. Any complete Freebase Virtuoso endpoint works for the repair —
# only standard ns-prefixed Freebase triples are queried.
#
# Modes:
#   bash scripts/setup_freebase_virtuoso.sh download [DIR]   # fetch + unzip dump, start service
#   bash scripts/setup_freebase_virtuoso.sh wait             # assume endpoint is up; poll it
#
# Download sources (53G+ disk; >=100GB RAM recommended):
#   Dropbox: https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
#   Baidu:   https://pan.baidu.com/s/1F0ytk74p8PGQ0tgAMu9--g?pwd=cp1j
#   Guide:   https://github.com/LHRLAB/ChatKBQA  (Freebase Virtuoso setup section)
set -e

MODE="${1:-wait}"
PORT="${VIRTUOSO_PORT:-3001}"
DIR="${2:-$HOME/virtuoso_db}"
ENDPOINT="http://localhost:${PORT}/sparql"

wait_endpoint() {
  echo "[wait] polling ${ENDPOINT} until it answers a Freebase query..."
  for i in $(seq 1 240); do
    ANS=$(curl -s --max-time 10 -G "${ENDPOINT}" \
          --data-urlencode 'query=SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } LIMIT 1' \
          -H "Accept: application/sparql-results+json" 2>/dev/null || true)
    if echo "$ANS" | grep -q '"value"'; then
      echo "[ready] Virtuoso SPARQL is up at ${ENDPOINT}"
      echo "[hint] export VIRTUOSO_ENDPOINT=${ENDPOINT} before running the repair script"
      exit 0
    fi
    sleep 5
  done
  echo "[TIMEOUT] endpoint did not answer in 20 min." >&2
  exit 1
}

case "$MODE" in
  wait)
    wait_endpoint
    ;;
  download)
    mkdir -p "$DIR"
    if [ ! -d "$DIR/virtuoso_db" ] && [ ! -f "$DIR/virtuoso.db" ]; then
      echo "[download] fetching virtuoso_db.zip (53G+) into $DIR ..."
      python3 - "$DIR" <<'PY'
import os, sys, urllib.request
d = sys.argv[1]
url = "https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip?dl=1"
urllib.request.urlretrieve(url, os.path.join(d, "virtuoso_db.zip"))
PY
      echo "[download] unzipping (takes a while)..."
      unzip -q "$DIR/virtuoso_db.zip" -d "$DIR"
    fi
    # Service wrapper from dki-lab/Freebase-Setup (bundled with ChatKBQA)
    if [ ! -f "$DIR/virtuoso.py" ]; then
      echo "[fetch] virtuoso.py service wrapper (dki-lab/Freebase-Setup)..."
      curl -sL -o "$DIR/virtuoso.py" \
        https://raw.githubusercontent.com/dki-lab/Freebase-Setup/master/virtuoso.py
    fi
    echo "[start] python3 $DIR/virtuoso.py start ${PORT} -d $DIR/virtuoso_db"
    python3 "$DIR/virtuoso.py" start "$PORT" -d "$DIR/virtuoso_db"
    wait_endpoint
    ;;
  *)
    echo "usage: $0 [download [DIR] | wait]" >&2
    exit 2
    ;;
esac
