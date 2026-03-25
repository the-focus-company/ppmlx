#!/usr/bin/env bash
# Benchmark: ppmlx + gpt-oss:20b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "gpt-oss:20b" "ppmlx" \
    "http://localhost:6767/v1/chat/completions" \
    "ppmlx:gpt-oss:20b"
