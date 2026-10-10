#!/usr/bin/env bash
# Multi-vendor OLMo experiment: run one cohort across seeds under the frozen
# spec, verdict each cell, stop on the first cell that says stop.
#
# The spec is `.design/multi-vendor-olmo-experiment.md`. This script exists
# so the band (all-NVIDIA, exa + 2 Pascal), the swap arm (AMD replaces a
# Pascal) and the add arm (AMD joins the full home cohort) are produced by
# ONE invocation shape where the only thing that differs is who dials in.
# Every training argument is built once here and appended to the controller
# invocation AND to every dial, so a walk-in can never train a different
# run than the one recorded. Run-in-Accept hands every admitted box the
# controller's own list at admission anyway, so the dial's copy only feeds
# the pre-dial model-signature probe.
#
# A consequence worth knowing: the controller's `--output` TRAVELS with
# that list, so a walk-in resolves the controller's relative run dir
# against its OWN cwd. A container dial must therefore run from the
# project dir the controller uses (`-w /workspace/ddp-bench` for the
# cuda-rank container, which shares exa's bind mount, so its rank
# artifacts land in the real cell and get rotated aside like a fan-out
# rank's), and because that container runs as root the cell is chowned
# back through it after each run. A read-only cwd (pascal's /mnt/rdl)
# only costs "Read-only file system" shipper warnings; the controller's
# own timeline carries every rank's samples regardless.
#
# Orchestration mirrors overshoot-sweep.sh, which proved it on this rig:
# open the window, dial the boxes, wait for the roster, fire, collect the
# controller and agent logs, judge. Deliberately a copy rather than a shared
# library: the two scripts move at different speeds and a sweep must stay
# reproducible from its own file.
#
#   COHORT=band  FARM=rig  MV_RANKS=3  MV_WALKINS="<dial>\n<dial>" ./mv-arms.sh
#
# Required:
#   COHORT       label of the cohort under test: band | swap | add | <probe>
#   FARM         the farm overlay (fdl.<FARM>.yml) the controller opens
#   MV_RANKS     ranks the cohort must reach before `fdl start` fires. The
#                farm's quorum is a floor, not the cohort: an AMD box that
#                is still dialing must not be left out of a 3-rank cell.
#   MV_WALKINS   one dial command per box that this script kicks, newline
#                separated, WITHOUT a trailing `-- <args>`. A box that dials
#                on its own (`persist: true` on a rented box) needs no entry;
#                it still counts in MV_RANKS. `$TOK` is exported from the
#                farm overlay before the dials run.
# Optional:
#   SEEDS        default "42 43 44 45 46": arms use the band's seeds so each
#                arm cell pairs with a band member
#   MV_TOKENS    default 20M; MV_SPLITS default 20. Overridable for probes
#                only; a cell that changes them is not the frozen spec.
#   MV_FDL_FLAGS default "-v": the controller's verbose tier carries the
#                per-reduce overshoot budget and the per-eval step, both of
#                which the verdict reads. Do not lower it.
#   MV_MONITOR   live dashboard port on the controller (default 8787). The
#                launcher binds it, `fdl status` prints it, and `fdl ui`'s run
#                tab embeds it; set empty to run without a live page.
#   MV_BAND      band root the verdict compares against (default runs/mv/band)
#   MV_STOP_ON   stop (default) exits 3 after a STOP verdict; continue runs on
#   MV_OUT       output root (default runs/mv/$COHORT)
#
# Cells land at ddp-bench/runs/mv/<COHORT>/s<seed>/olmo-graph/cpu-async-diloco/
# with controller.log, walkin-N.log, elche-config.txt, provenance.txt and
# verdict.txt beside the run's own artifacts. Resume-safe: a cell whose
# training.log carries an `epoch ` line and the `# total:` footer is skipped.
set -u
cd "$(dirname "$0")/.."

COHORT=${COHORT:?set COHORT (band | swap | add | <probe label>)}
FARM=${FARM:?set FARM (the overlay fdl.<FARM>.yml the controller opens)}
MV_RANKS=${MV_RANKS:?set MV_RANKS (ranks the cohort must reach before start)}
SEEDS=${SEEDS:-42 43 44 45 46}
MV_TOKENS=${MV_TOKENS:-20M}
MV_SPLITS=${MV_SPLITS:-20}
MV_FDL_FLAGS=${MV_FDL_FLAGS:--v}
MV_MONITOR=${MV_MONITOR-8787}
MV_BAND=${MV_BAND:-runs/mv/band}
MV_STOP_ON=${MV_STOP_ON:-stop}
OUT=${MV_OUT:-runs/mv/$COHORT}
ABS_OUT=ddp-bench/$OUT
LOGDIR=${MV_LOGDIR:-target/mv-arms/$COHORT}
MODEL=olmo-graph
CELL=$MODEL/cpu-async-diloco
mkdir -p "$ABS_OUT" "$LOGDIR"

# THE FROZEN SPEC. The 08-13 baseline invocation (overshoot sweep arm A)
# plus the two things the experiment added on 2026-09-20: a consensus eval
# per split and the in-domain held-out slice. Nothing else moves between
# the band and the arms.
FIXED="--model $MODEL --mode cpu-async --outer-optimizer diloco --train-tokens $MV_TOKENS --bf16-wire --epochs 1 --epoch-splits $MV_SPLITS --per-epoch-eval --olmo-eval in-domain --save-dashboard"

ts() { date '+%F %T'; }

cell_done() {
  log="$ABS_OUT/$1/$CELL/training.log"
  grep -q '^epoch ' "$log" 2>/dev/null && grep -q '# total:' "$log" 2>/dev/null
}

strays() { pgrep -af 'release/ddp-benc[h]' >/dev/null 2>&1; }

# libtorch `[W...] Warning:` lines do not fail a run, and a ladder whose
# green summary hides them is worse than no ladder.
warns_in() { grep -cE '\[W[0-9]* ' "$1" 2>/dev/null || true; }

# The identity gate: the run itself is asked which knobs it used rather
# than trusted to have received them. The harness echoes its effective
# elche config and the log header names the eval set and the cohort.
identity_ok() {
  # $1 controller log, $2 training log
  line=$(grep -m1 '  elche:' "$1" 2>/dev/null)
  if [ -z "$line" ]; then echo "no elche config echo in the controller log"; return 1; fi
  case "$line" in
    *"max_overshoot=auto "*) ;;
    *) echo "effective max_overshoot != auto | $line"; return 1 ;;
  esac
  case "$line" in
    *"easgd_alpha=Some(0.5)"*) ;;
    *) echo "effective easgd_alpha != 0.5 | $line"; return 1 ;;
  esac
  if ! grep -q '^# eval: in-domain' "$2" 2>/dev/null; then
    echo "training.log header does not name the in-domain eval"; return 1
  fi
  got=$(grep -c '^# gpu r' "$2" 2>/dev/null || true)
  if [ "$got" -ne "$MV_RANKS" ]; then
    echo "cohort had $got rank(s) in the log header, expected $MV_RANKS"; return 1
  fi
  return 0
}

# Reap a failed cell's controller and WAIT for its port to come back. The
# launcher is fdl's grandchild and the thing holding 1337; a containerised
# controller is PID 1 of its own namespace, so the container is what goes.
reap_controller() {
  kill "$@" 2>/dev/null
  sleep 2
  for _ in 1 2 3 4 5; do
    holder=$(ss -ltnp 2>/dev/null | sed -n 's/.*:1337 .*pid=\([0-9]*\).*/\1/p' | head -1)
    [ -z "$holder" ] && return 0
    cid=$(grep -oE 'docker[-/][0-9a-f]{12}' "/proc/$holder/cgroup" 2>/dev/null \
            | head -1 | grep -oE '[0-9a-f]{12}')
    if [ -n "$cid" ]; then docker rm -f "$cid" >/dev/null 2>&1; else kill -9 "$holder" 2>/dev/null; fi
    sleep 2
  done
  if ss -ltn 2>/dev/null | grep -q ':1337 '; then
    echo "$(ts) WARN: port 1337 is still held; the next cell will fail to bind."
  fi
}

# Provenance, written from the success path only. Beyond the sweep's
# stamp: the roster as the controller admitted it (host, ranks, GPU count,
# libtorch label -- the `rocm` label in that line IS the demo), the dials
# with their credential redacted, and the digest of the tree this box
# serves. The digest binds only walk-ins that pulled source through the
# door; a `--bin` walk-in ran the binary its dial names, and the stamp says
# which case each dial is.
stamp() {
  # $1 label, $2 seed, $3 invocation, $4 walk-in count, $5 controller log
  d="$ABS_OUT/$1/$CELL"; mkdir -p "$d"
  run_tree="${FLODL_HOME:-$HOME/.flodl}/run/tree"
  digest=$(sed -n 's/^ *sha256: *//p' "$run_tree/.fdl-run.yml" 2>/dev/null | head -1)
  { echo "cohort:         $COHORT"
    echo "seed:           $2"
    echo "ranks:          $MV_RANKS"
    echo "rank0:          by dial order, not by pace (the controller's Fastest role election is rank 0 in practice; TODO.md)"
    echo "model:          $MODEL"
    echo "spec:           $FIXED"
    echo "topology:       walk-in (farm=$FARM), rental-parity"
    echo "invocation:     $3"
    echo "walkins:        $4 box(es) kicked by this script"
    printf '%s\n' "$MV_WALKINS" | sed -E 's/(--token[= ])[^ ]+/\1<redacted>/g' | sed 's/^/dial:           /'
    grep -a 'joined with ranks' "$5" 2>/dev/null | sed 's/^.*host /roster:         host /'
    if [ -n "$digest" ]; then
      echo "served_tree_sha256: $digest (binds source-pull walk-ins only; a --bin dial ran the binary it names)"
    fi
    echo "git_sha:        $(git rev-parse HEAD 2>/dev/null)"
    echo "git_dirty:      $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
    echo "utc:            $(date -u '+%F %T UTC')"
    echo "host:           $(hostname)"
    echo "libtorch:       $(cat libtorch/.active 2>/dev/null || echo unknown)"
  } > "$d/provenance.txt"
}

if [ ! -f "fdl.$FARM.yml" ]; then
  echo "$(ts) ABORT: farm overlay fdl.$FARM.yml not found (it is user-local; create it with fdl join-config $FARM)"
  exit 1
fi
MV_WALKINS=${MV_WALKINS:-}

# The farm's RESOLVED token (an overlay may inherit it), exported for dials
# that reference $TOK. The file grep is the fallback.
TOK=$(./fdl "@$FARM" config show 2>/dev/null \
        | sed -n "s/^ *token: *//p" | head -1 | awk '{print $1}' | tr -d "\"' \r")
if [ -z "$TOK" ]; then
  TOK=$(sed -n "s/^ *token: *//p" "fdl.$FARM.yml" | head -1 | tr -d "\"' \r")
fi
export TOK

echo "$(ts) MV ARMS BEGIN cohort=$COHORT farm=$FARM ranks=$MV_RANKS rev=$(git rev-parse --short HEAD) out=$OUT"
echo "$(ts) spec: $FIXED"
echo "$(ts) seeds: $SEEDS"

for seed in $SEEDS; do
  label="s$seed"
  if cell_done "$label"; then echo "$(ts) SKIP $label (done)"; continue; fi
  echo "$(ts) START $COHORT/$label"

  CORE_ARGS="$FIXED --seed $seed"
  clog="$LOGDIR/$label-controller.log"

  # The live dashboard is the controller's: the flag rides the controller
  # line only (it reaches the walk-ins through the accept anyway, where a
  # rank child never binds it).
  monitor_arg=""
  [ -n "$MV_MONITOR" ] && monitor_arg="--monitor $MV_MONITOR"
  # shellcheck disable=SC2086
  ./fdl $MV_FDL_FLAGS "@$FARM" ddp-bench $CORE_ARGS $monitor_arg --output "$OUT/$label" > "$clog" 2>&1 &
  cpid=$!

  waited=0
  until grep -aq "join: window open" "$clog" 2>/dev/null; do
    sleep 2; waited=$((waited+2))
    if [ "$waited" -ge 180 ] || ! kill -0 "$cpid" 2>/dev/null; then
      echo "$(ts) FAIL $label: controller never opened a join window"
      tail -15 "$clog" | sed 's/^/    /'
      reap_controller "$cpid"; wait "$cpid" 2>/dev/null
      exit 1
    fi
  done

  # Dials are kicked IN ORDER, each waiting for its admission before the
  # next one goes: ranks are numbered in admission order, and the
  # coordinator's role election (eval, checkpoint, epoch callback) settles
  # on the lowest live rank before any pace is known and stays there, so a
  # dial race decides which box runs the evals. A Pascal winning the race
  # cost the probe 3x per eval and 22% of wall (2026-09-20). List the
  # fastest box first.
  wpids=""; wlogs=""; nw=0
  while IFS= read -r cmd; do
    [ -n "$cmd" ] || continue
    case "$cmd" in
      *" -- "*)
        echo "$(ts) FAIL $label: an MV_WALKINS entry carries its own '--'; args are appended here so the spec cannot drift"
        # shellcheck disable=SC2086
        reap_controller "$cpid" $wpids; wait 2>/dev/null
        exit 1 ;;
    esac
    before=$(./fdl "@$FARM" status 2>/dev/null | sed -n 's/^ *ranks: *\([0-9]*\) joined.*/\1/p' | head -1)
    [ -n "$before" ] || before=0
    nw=$((nw+1))
    wlog="$LOGDIR/$label-walkin-$nw.log"
    # stdin from /dev/null: a backgrounded ssh dial otherwise inherits this
    # loop's heredoc as its stdin and swallows every dial line after it. The
    # band never met it because its ssh dial was listed last; a pascal-first
    # probe lost exa's dial to it (2026-09-20).
    sh -c "$cmd -- $CORE_ARGS" < /dev/null > "$wlog" 2>&1 &
    wpids="$wpids $!"
    wlogs="$wlogs $wlog"
    waited=0
    while :; do
      now=$(./fdl "@$FARM" status 2>/dev/null | sed -n 's/^ *ranks: *\([0-9]*\) joined.*/\1/p' | head -1)
      [ -n "$now" ] || now=0
      [ "$now" -gt "$before" ] && break
      sleep 2; waited=$((waited+2))
      if [ "$waited" -ge 180 ] || ! kill -0 "$cpid" 2>/dev/null; then
        echo "$(ts) FAIL $label: dial $nw was not admitted within 180s"
        tail -8 "$wlog" | sed 's/^/    /'
        # shellcheck disable=SC2086
        reap_controller "$cpid" $wpids; wait 2>/dev/null
        exit 1
      fi
    done
    echo "$(ts) $label: dial $nw admitted ($now rank(s) in)"
  done <<EOF
$MV_WALKINS
EOF
  echo "$(ts) $label: $nw box(es) in, waiting for $MV_RANKS rank(s)"

  # Wait for the WHOLE cohort, not the farm's quorum: `fdl status` prints
  # `ranks: N joined` and, at quorum, the startable line. Both must hold.
  waited=0
  while :; do
    st=$(./fdl "@$FARM" status 2>/dev/null)
    joined=$(printf '%s\n' "$st" | sed -n 's/^ *ranks: *\([0-9]*\) joined.*/\1/p' | head -1)
    [ -n "$joined" ] || joined=0
    if [ "$joined" -ge "$MV_RANKS" ] && printf '%s\n' "$st" | grep -q "roster startable"; then break; fi
    sleep 3; waited=$((waited+3))
    if [ "$waited" -ge 600 ] || ! kill -0 "$cpid" 2>/dev/null; then
      echo "$(ts) FAIL $label: cohort never reached $MV_RANKS rank(s) (last seen: $joined)"
      printf '%s\n' "$st" | sed 's/^/    /' | tail -8
      # shellcheck disable=SC2086
      reap_controller "$cpid" $wpids; wait 2>/dev/null
      exit 1
    fi
  done
  ./fdl "@$FARM" start >> "$clog" 2>&1

  wait "$cpid"; rc=$?
  # shellcheck disable=SC2086
  wait $wpids 2>/dev/null

  # The cuda-rank walk-in runs as root and writes into the cell through the
  # shared bind mount; hand the cell back to this user through the same
  # container (the documented recipe; host sudo is not the answer).
  docker exec rdl-cuda-rank-1 chown -R "$(id -u):$(id -g)" "/workspace/$ABS_OUT/$label" >/dev/null 2>&1 || true

  degraded=$(grep -c "finished DEGRADED\|child exit(s) tolerated\|device-side assert" "$clog")
  warns=$(warns_in "$clog")
  mkdir -p "$ABS_OUT/$label/$CELL"
  grep -m1 '  elche:' "$clog" > "$ABS_OUT/$label/$CELL/elche-config.txt" 2>/dev/null
  cp "$clog" "$ABS_OUT/$label/$CELL/controller.log" 2>/dev/null

  # A walk-in agent owns its own stdout, so a rank-side libtorch warning
  # never reaches the controller log; count across every agent log too.
  agents_ok=1
  for wlog in $wlogs; do
    if ! grep -aq "finished cleanly" "$wlog"; then
      echo "$(ts) $label: walk-in agent did not finish cleanly ($wlog)"
      tail -12 "$wlog" | sed 's/^/    /'
      agents_ok=0
    fi
    wwarn=$(warns_in "$wlog")
    warns=$((warns + wwarn))
    cp "$wlog" "$ABS_OUT/$label/$CELL/$(basename "$wlog" | sed "s/^$label-//")" 2>/dev/null
  done

  tlog="$ABS_OUT/$label/$CELL/training.log"
  if id_err=$(identity_ok "$clog" "$tlog"); then id_ok=1; else id_ok=0; fi

  if [ $rc -eq 0 ] && [ "$degraded" -eq 0 ] && [ "$warns" -eq 0 ] \
     && [ "$agents_ok" -eq 1 ] && [ "$id_ok" -eq 1 ] \
     && grep -aq "done:" "$clog" && cell_done "$label"; then
    stamp "$label" "$seed" "fdl $MV_FDL_FLAGS @$FARM ddp-bench $CORE_ARGS $monitor_arg --output $OUT/$label" "$nw" "$clog"
    echo "$(ts) OK $COHORT/$label"
  else
    echo "$(ts) FAIL $COHORT/$label rc=$rc degraded=$degraded libtorch_warnings=$warns agents_ok=$agents_ok identity=$id_ok"
    [ "$id_ok" -eq 1 ] || echo "$(ts)   identity: $id_err"
    # Keep the logs for the post-mortem, drop the run artifacts so the cell
    # is not mistaken for data and re-runs on the next pass.
    keep="$LOGDIR/failed-$label-$(date -u '+%Y%m%dT%H%M%S')"
    mkdir -p "$keep"
    mv "$ABS_OUT/$label/$CELL"/*.log "$keep/" 2>/dev/null
    rm -rf "${ABS_OUT:?}/$label"
    sleep 15
    if strays; then
      echo "$(ts) ABORT-DIRTY: leftover ddp-bench processes after the failed cell; clean up before re-running:"
      pgrep -af 'release/ddp-benc[h]'
      exit 1
    fi
    echo "$(ts) STOP: a failed cell is a stop by definition. Stop spending and decide: investigate, or MV_STOP_ON=continue."
    exit 3
  fi

  # The verdict card: clean or not, in the band or not, d_raw against the
  # guard floor, the reduce count and the eval curve with its steps.
  if command -v python3 >/dev/null 2>&1; then
    # The band root is passed for band cells too: the verdict then places
    # the cell among the OTHER members (leave-one-out) and pairs nothing.
    python3 ddp-bench/scripts/mv_verdict.py "$ABS_OUT/$label/$CELL" --seed "$seed" --band "ddp-bench/$MV_BAND" \
      | tee "$ABS_OUT/$label/$CELL/verdict.txt"
    vrc=${PIPESTATUS[0]}
    if [ "$vrc" -eq 3 ] && [ "$MV_STOP_ON" = stop ]; then
      echo "$(ts) STOP after $COHORT/$label: the verdict says stop. This ladder destroys nothing; stop spending and decide (investigate, MV_STOP_ON=continue, or the destroy runbook if the box is rented)."
      exit 3
    fi
  fi
  # Settle between cells: rank-0 SIGSEGV at CUDA init was observed on cells
  # starting within ~1s of the previous teardown.
  sleep 10
done

echo "$(ts) MV ARMS DONE cohort=$COHORT"
