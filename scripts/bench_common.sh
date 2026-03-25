#!/usr/bin/env bash
# bench_common.sh — shared functions for ppmlx vs ollama benchmarks
# Source this file, don't run it directly.

RUNS=3
MAX_TOKENS=512
RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmark_results"
mkdir -p "$RESULTS_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Prompts ───────────────────────────────────────────────────────────────

SIMPLE_PROMPT="Explain the difference between TCP and UDP in exactly 3 bullet points. Be precise and technical."

COMPLEX_PROMPT="Design a rate limiter for an API gateway. Describe:
1. The algorithm you'd choose and why (token bucket vs sliding window vs fixed window)
2. Data structures needed
3. How to handle distributed deployment (multiple gateway instances)
4. Edge cases (clock skew, burst traffic, graceful degradation)
Provide pseudocode for the core logic."

AGENTIC_PROMPT="Review this Python function. Find all bugs and security issues, explain each one, then provide a corrected version with inline comments explaining every fix.

\`\`\`python
def process_user_data(user_input):
    import os
    query = f\"SELECT * FROM users WHERE name = '{user_input}'\"
    result = os.popen(f'echo {user_input} | base64').read()
    data = eval(result)
    return data
\`\`\`"

# ── API call helper (streaming — measures TTFT) ──────────────────────────
#
# Uses streaming mode to measure time-to-first-token accurately.
# Parses SSE chunks, captures first content delta timestamp.
# Returns JSON: {ms, tokens, tok_s, ttft_ms, error}

call_api() {
    local url="$1"
    local model="$2"
    local prompt="$3"
    local label="$4"

    local body
    body=$(jq -n \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{
            model: $model,
            messages: [{"role": "user", "content": $prompt}],
            max_tokens: $max_tokens,
            temperature: 0.0,
            stream: true
        }')

    # Use python to stream and measure TTFT precisely
    python3 -c "
import sys, json, time, urllib.request

url = sys.argv[1]
data = sys.argv[2].encode()
label = sys.argv[3]

req = urllib.request.Request(url, data=data, headers={
    'Content-Type': 'application/json',
    'Authorization': 'Bearer local',
})

start_ns = time.time_ns()
ttft_ns = None
chunks = 0
full_text = []

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = b''
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b'\n\n' in buf:
                line, buf = buf.split(b'\n\n', 1)
                line = line.decode('utf-8', errors='replace').strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    break
                try:
                    obj = json.loads(payload)
                    delta = obj.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        if ttft_ns is None:
                            ttft_ns = time.time_ns()
                        chunks += 1
                        full_text.append(content)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
except Exception as e:
    print(json.dumps({'ms':0,'tokens':0,'tok_s':0,'ttft_ms':0,'error':True}))
    sys.exit(0)

end_ns = time.time_ns()
elapsed_ms = round((end_ns - start_ns) / 1_000_000)
ttft_ms = round((ttft_ns - start_ns) / 1_000_000) if ttft_ns else 0

# Estimate token count from chunks (each SSE chunk ≈ 1 token for most backends)
tokens = chunks
tok_s = round(tokens / (elapsed_ms / 1000), 2) if elapsed_ms > 0 and tokens > 0 else 0

print(json.dumps({
    'ms': elapsed_ms,
    'tokens': tokens,
    'tok_s': tok_s,
    'ttft_ms': ttft_ms,
    'error': False,
}))
" "$url" "$body" "$label" 2>/dev/null
}

# ── Peak memory helper ────────────────────────────────────────────────────
#
# Reads GPU/unified memory from the backend's health/status endpoint.
# ppmlx: /health → .system.memory_total_gb (and loaded models)
# ollama: /api/ps → loaded model VRAM

get_memory_mb() {
    local backend="$1"

    if [[ "$backend" == "ppmlx" ]]; then
        # ppmlx /health reports loaded models
        local health
        health=$(curl -s --max-time 5 "http://localhost:6767/health" 2>/dev/null) || echo "0"
        # Get loaded model memory from ps-like info if available
        local loaded
        loaded=$(echo "$health" | jq -r '.loaded_models | length // 0' 2>/dev/null) || loaded=0
        # Use system_profiler for actual GPU memory pressure
        local mem_pressure
        mem_pressure=$(memory_pressure 2>/dev/null | grep "System-wide memory free percentage:" | awk '{print $NF}' | tr -d '%') || mem_pressure=""
        if [[ -n "$mem_pressure" ]]; then
            echo "$mem_pressure"
        else
            echo "0"
        fi
    elif [[ "$backend" == "ollama" ]]; then
        local ps
        ps=$(curl -s --max-time 5 "http://localhost:11434/api/ps" 2>/dev/null) || echo "0"
        local vram
        vram=$(echo "$ps" | jq -r '.models[0].size_vram // 0' 2>/dev/null) || vram=0
        echo $(python3 -c "print(round($vram / 1048576))") # bytes → MB
    else
        echo "0"
    fi
}

# ── Pi CLI call helper ────────────────────────────────────────────────────

call_pi() {
    local pi_model="$1"
    local prompt="$2"
    local label="$3"

    local start_ns end_ns output
    start_ns=$(python3 -c "import time; print(int(time.time_ns()))")

    output=$(pi --model "$pi_model" "$prompt" 2>/dev/null) || true

    end_ns=$(python3 -c "import time; print(int(time.time_ns()))")

    local elapsed_ms
    elapsed_ms=$(python3 -c "print(round(($end_ns - $start_ns) / 1_000_000))")

    local char_count=0
    if [[ -n "$output" ]]; then
        char_count=${#output}
    fi

    if [[ "$char_count" -eq 0 ]]; then
        printf "${RED}    [%s] pi returned empty output${RESET}\n" "$label" >&2
        echo "{\"ms\":0,\"chars\":0,\"error\":true}"
        return
    fi

    echo "{\"ms\":$elapsed_ms,\"chars\":$char_count,\"error\":false}"
}

# ── Run a single scenario (direct API, streaming) ────────────────────────

run_direct() {
    local scenario_name="$1"
    local prompt="$2"
    local api_url="$3"
    local model="$4"
    local backend="$5"

    printf "\n  ${BOLD}%s${RESET}\n" "$scenario_name"

    local total_ms=0 total_tok=0 total_tps=0 total_ttft=0 ok=0

    for i in $(seq 1 $RUNS); do
        printf "    ${DIM}run %d/%d...${RESET}\r" "$i" "$RUNS"
        local result
        result=$(call_api "$api_url" "$model" "$prompt" "$backend")

        local err ms tok tps ttft
        err=$(echo "$result" | jq -r '.error')
        if [[ "$err" == "false" ]]; then
            ms=$(echo "$result" | jq -r '.ms')
            tok=$(echo "$result" | jq -r '.tokens')
            tps=$(echo "$result" | jq -r '.tok_s')
            ttft=$(echo "$result" | jq -r '.ttft_ms')
            total_ms=$((total_ms + ms))
            total_tok=$((total_tok + tok))
            total_tps=$(python3 -c "print($total_tps + $tps)")
            total_ttft=$((total_ttft + ttft))
            ok=$((ok + 1))
        fi
    done

    if [[ "$ok" -gt 0 ]]; then
        local avg_ms=$((total_ms / ok))
        local avg_tps
        avg_tps=$(python3 -c "print(round($total_tps / $ok, 1))")
        local avg_tok=$((total_tok / ok))
        local avg_ttft=$((total_ttft / ok))
        printf "    ${GREEN}%dms total  |  %dms TTFT  |  %s tok/s  |  %d tokens  (%d/%d)${RESET}\n" \
            "$avg_ms" "$avg_ttft" "$avg_tps" "$avg_tok" "$ok" "$RUNS"
        echo "$avg_ms $avg_tps $avg_tok $avg_ttft" > "$RESULTS_DIR/.last_direct"
    else
        printf "    ${RED}all runs failed${RESET}\n"
        echo "0 0 0 0" > "$RESULTS_DIR/.last_direct"
    fi
}

# ── Run a single scenario (pi CLI) ───────────────────────────────────────

run_pi() {
    local scenario_name="$1"
    local prompt="$2"
    local pi_model="$3"
    local backend="$4"

    printf "\n  ${BOLD}%s (via pi)${RESET}\n" "$scenario_name"

    local total_ms=0 total_chars=0 ok=0

    for i in $(seq 1 $RUNS); do
        printf "    ${DIM}run %d/%d...${RESET}\r" "$i" "$RUNS"
        local result
        result=$(call_pi "$pi_model" "$prompt" "$backend")

        local err ms chars
        err=$(echo "$result" | jq -r '.error')
        if [[ "$err" == "false" ]]; then
            ms=$(echo "$result" | jq -r '.ms')
            chars=$(echo "$result" | jq -r '.chars')
            total_ms=$((total_ms + ms))
            total_chars=$((total_chars + chars))
            ok=$((ok + 1))
        fi
    done

    if [[ "$ok" -gt 0 ]]; then
        local avg_ms=$((total_ms / ok))
        local avg_chars=$((total_chars / ok))
        printf "    ${GREEN}%dms avg  |  %d chars  (%d/%d)${RESET}\n" \
            "$avg_ms" "$avg_chars" "$ok" "$RUNS"
        echo "$avg_ms $avg_chars" > "$RESULTS_DIR/.last_pi"
    else
        printf "    ${RED}all runs failed${RESET}\n"
        echo "0 0" > "$RESULTS_DIR/.last_pi"
    fi
}

# ── Full benchmark for one model+backend ──────────────────────────────────

run_full_benchmark() {
    local model="$1"
    local backend="$2"
    local api_url="$3"
    local pi_model="$4"

    local outfile="$RESULTS_DIR/${backend}_${model//[:\/]/_}.json"
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf "\n${BOLD}════════════════════════════════════════${RESET}\n"
    printf "${BOLD}  %s — %s${RESET}\n" "$backend" "$model"
    printf "${BOLD}════════════════════════════════════════${RESET}\n"
    printf "Runs: %d | Max tokens: %d\n" "$RUNS" "$MAX_TOKENS"

    # Measure memory before
    local mem_before
    mem_before=$(get_memory_mb "$backend")

    # Scenario 1: Simple (direct, streaming)
    run_direct "Scenario 1: Simple (direct)" "$SIMPLE_PROMPT" "$api_url" "$model" "$backend"
    read -r s1_ms s1_tps s1_tok s1_ttft < "$RESULTS_DIR/.last_direct"

    # Scenario 2: Complex (direct, streaming)
    run_direct "Scenario 2: Complex (direct)" "$COMPLEX_PROMPT" "$api_url" "$model" "$backend"
    read -r s2_ms s2_tps s2_tok s2_ttft < "$RESULTS_DIR/.last_direct"

    # Measure memory after model is loaded
    local mem_after
    mem_after=$(get_memory_mb "$backend")

    # Scenario 3: Agentic (via pi)
    run_pi "Scenario 3: Agentic" "$AGENTIC_PROMPT" "$pi_model" "$backend"
    read -r s3_ms s3_chars < "$RESULTS_DIR/.last_pi"

    # Write results JSON
    cat > "$outfile" <<JSONEOF
{
  "backend": "$backend",
  "model": "$model",
  "timestamp": "$timestamp",
  "runs": $RUNS,
  "max_tokens": $MAX_TOKENS,
  "memory_mb": $mem_after,
  "results": {
    "simple":  { "avg_ms": $s1_ms, "avg_tok_s": $s1_tps, "avg_tokens": $s1_tok, "avg_ttft_ms": $s1_ttft },
    "complex": { "avg_ms": $s2_ms, "avg_tok_s": $s2_tps, "avg_tokens": $s2_tok, "avg_ttft_ms": $s2_ttft },
    "agentic": { "avg_ms": $s3_ms, "avg_chars": $s3_chars }
  }
}
JSONEOF

    printf "\n${GREEN}Results saved → %s${RESET}\n" "$outfile"
    rm -f "$RESULTS_DIR/.last_direct" "$RESULTS_DIR/.last_pi"
}
