#!/usr/bin/env bash
# Benchmark: ollama + gpt-oss:20b
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bench_common.sh"

run_full_benchmark "gpt-oss:20b" "ollama" \
    "http://localhost:11434/v1/chat/completions" \
    "ollama:gpt-oss:20b"
