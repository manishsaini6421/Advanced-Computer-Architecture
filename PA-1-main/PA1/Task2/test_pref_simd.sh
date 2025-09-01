#!/bin/bash

# Set core affinity for the script’s process itself (can be adapted if specific pinning is desired later)
taskset -cp 2,3 $$ >/dev/null

# --- Configurable parameters (Extend as needed) ---
EXEC="./simdpref"   # C++ binary built from emb.cpp
OUTPUT_DIR="perf_simd_pref"

# These arrays should be expanded as necessary
EMBED_SIZES=(2000000)
PREFETCH_DISTANCES=(2 4 8 16)
CACHE_LEVELS=("L1" "L2" "LLC")  # Map to prefetch hints
SOFTWARE_PREFETCH=("on" "off")
EMBED_DIMS=(128 512 1024)
SIMD_WIDTHS=("sse" "avx")

# Perf events relevant for your table
PERF_EVENTS="L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses"
# Perf events (generic + requested). You can override by exporting PERF_EVENTS before running.
# Includes requested: L1-dcache-load-misses,L2_rqsts.miss,LLC-load-misses,sw_prefetch_access.t0
: ${PERF_EVENTS:="cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,L1-dcache-load-misses,L2_rqsts.miss,LLC-load-misses"}
PERF_FLAGS="--all-user --no-big-num"
# PERF_EVENTS="L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses,sw_prefetch_access.t0"

mkdir -p "$OUTPUT_DIR"

# Detect if perf is usable (handles perf_event_paranoid restrictions)
USE_PERF=1
if ! perf stat -e cycles true >/dev/null 2>&1; then
  echo "perf not usable in this environment (likely due to perf_event_paranoid). Proceeding without perf." >&2
  USE_PERF=0
fi

# Validate requested perf events and keep only supported ones
PERF_EVENTS_JOINED=""
if [ "$USE_PERF" -eq 1 ]; then
  IFS=',' read -r -a REQ_EVENTS <<< "$PERF_EVENTS"
  VALID_EVENTS=()
  for ev in "${REQ_EVENTS[@]}"; do
    ev_trimmed="${ev## }"; ev_trimmed="${ev_trimmed%% }"
    if [ -n "$ev_trimmed" ] && perf stat -e "$ev_trimmed" true >/dev/null 2>&1; then
      VALID_EVENTS+=("$ev_trimmed")
    else
      echo "Skipping unsupported perf event: $ev_trimmed" >&2
    fi
  done
  if [ ${#VALID_EVENTS[@]} -gt 0 ]; then
    PERF_EVENTS_JOINED=$(IFS=,; echo "${VALID_EVENTS[*]}")
  else
    echo "No requested perf events are supported; running without perf." >&2
    USE_PERF=0
  fi
fi

prefetch_hint_flag() {
  case "$1" in
    "L1") echo "_MM_HINT_T0";;
    "L2") echo "_MM_HINT_T1";;
    "LLC") echo "_MM_HINT_T2";;
    *) echo "_MM_HINT_T0";;
  esac
}

prefetch_event_flag() {
  case "$1" in
    "L1") echo "sw_prefetch_access.t0";;
    "L2") echo "sw_prefetch_access.t1_t2";;
    "LLC") echo "sw_prefetch_access.nta";;
    *) echo "";;
  esac
}

# No parsing; we save raw perf+program output per run

for prefetch in "${SOFTWARE_PREFETCH[@]}"; do
# Main experiment loop
    for et_size in "${EMBED_SIZES[@]}"; do
      for dim in "${EMBED_DIMS[@]}"; do
       for simd in "${SIMD_WIDTHS[@]}"; do
        for pdist in "${PREFETCH_DISTANCES[@]}"; do
            for cache_lvl in "${CACHE_LEVELS[@]}"; do

                # Map cache level to prefetch hint if needed
                cache_hint=$(prefetch_hint_flag $cache_lvl)
                sw_event=$(prefetch_event_flag $cache_lvl)
                EVENTS_TO_USE="$PERF_EVENTS_JOINED"

                # if [ -n "$sw_event" ]; then
                # fi

                sw_event=$(prefetch_event_flag $cache_lvl)

                if [ -n "$sw_event" ]; then
                  EVENTS_TO_USE="$EVENTS_TO_USE,$sw_event"
                fi

                # Compose args for the embedding binary
                # --software-prefetch=[on|off] --embed-size=X --prefetch-distance=Y --prefetch-level=Z
                args="--software-prefetch=$prefetch --embed-size=$et_size --embed-dim=$dim  --simd-width=$simd --prefetch-distance=$pdist --prefetch-level=$cache_hint"
     
                out_file="$OUTPUT_DIR/out_prefetch-${prefetch}_size-${et_size}_dim-${dim}_simd-${simd}_dist-${pdist}_lvl-${cache_lvl}_try-${tr}.txt"

                echo $out_file;

                if [ "$USE_PERF" -eq 1 ]; then
                  echo "[perf events] $EVENTS_TO_USE" > "$out_file"
                  perf stat $PERF_FLAGS -e "$EVENTS_TO_USE" -- taskset -c 0,1 $EXEC $args >> "$out_file" 2>&1
                else
                  echo "[perf disabled] Running without perf stat due to permissions." > "$out_file"
                  taskset -c 0,1 $EXEC $args >> "$out_file" 2>&1
                fi
               
            done
        done
      done
      done 

      done
done