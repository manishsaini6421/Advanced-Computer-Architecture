#!/bin/bash

# Set core affinity for the script’s process itself (can be adapted if specific pinning is desired later)
taskset -cp 1-3,5-7 $$ >/dev/null

# --- Configurable parameters (Extend as needed) ---
EXEC="./embedding_test"   # Replace with your embedding operation binary
OUTPUT_FILE="embedding_perf_results.csv"

# These arrays should be expanded as necessary
EMBED_SIZES=(1000 1000000)
PREFETCH_DISTANCES=(4 16)
CACHE_LEVELS=("L1" "L2" "LLC")  # Map to prefetch hints
SOFTWARE_PREFETCH=("off" "on")

# Perf events relevant for your table
PERF_EVENTS="L1-dcache-load-misses,L2_rqsts.miss,LLC-load-misses,sw_prefetch_access"

# CSV header
echo "Prefetching,EmbedTableSize,PrefetchDistance,CacheLevel,L1D_misses,L2_misses,LLC_misses,SW_prefetch_requests,ExecTime_ms" > "$OUTPUT_FILE"

prefetch_hint_flag() {
  case "$1" in
    "L1") echo "_MM_HINT_T0";;
    "L2") echo "_MM_HINT_T1";;
    "LLC") echo "_MM_HINT_T2";;
    *) echo "_MM_HINT_T0";;
  esac
}

parse_perf_output() {
    # Usage: parse_perf_output "$perf_output"
    local po="$1"
    local l1d=$(echo "$po" | grep "L1-dcache-load-misses" | awk '{print $1}' | tr -d ',')
    local l2=$(echo "$po" | grep "L2_rqsts.miss"     | awk '{print $1}' | tr -d ',')
    local llc=$(echo "$po" | grep "LLC-load-misses"  | awk '{print $1}' | tr -d ',')
    local swp=$(echo "$po" | grep "sw_prefetch_access" | awk '{print $1}' | tr -d ',')
    echo "$l1d,$l2,$llc,$swp"
}

# Main experiment loop
for prefetch in "${SOFTWARE_PREFETCH[@]}"; do
    for et_size in "${EMBED_SIZES[@]}"; do
        for pdist in "${PREFETCH_DISTANCES[@]}"; do
            for cache_lvl in "${CACHE_LEVELS[@]}"; do

                # Map cache level to prefetch hint if needed
                cache_hint=$(prefetch_hint_flag $cache_lvl)

                # Compose args for the embedding binary
                # --software-prefetch=[on|off] --embed-size=X --prefetch-distance=Y --prefetch-level=Z
                args="--software-prefetch=$prefetch --embed-size=$et_size --prefetch-distance=$pdist --prefetch-level=$cache_hint"

                # Run perf (run multiple times if you want to average)
                for tr in {1..5}; do
                  perf_out=$(perf stat -e $PERF_EVENTS taskset -c 0,4 $EXEC $args 2>&1)
                  # Replace this grep with suitable extraction from your exec output:
                  exec_time=$(echo "$perf_out" | grep "embedding operation took" | awk '{print $4}')

                  # Fall back if Python/time not instrumented, e.g., use `time` or similar.
                  [ -z "$exec_time" ] && exec_time=$(echo "$perf_out" | grep "real" | awk '{print $2}')

                  perf_metrics=$(parse_perf_output "$perf_out")

                  # Compose row for CSV
                  echo "$prefetch,$et_size,$pdist,$cache_lvl,$perf_metrics,$exec_time" >> "$OUTPUT_FILE"
                done

            done
        done
    done
done
