#!/usr/bin/env bash
# One rented-GPU session (RunPod, 1x RTX 4090, PyTorch template), start to finish.
#
#   bash scripts/gpu_session.sh                   # the whole session
#   bash scripts/gpu_session.sh --skip-baseline   # rerun without step c (base Qwen is already done)
#   bash scripts/gpu_session.sh --dry-run         # no GPU: check paths, YAML, data; print the plan
#
# Steps, in order. Each logs to logs/<step>.log and the script stops on the first error:
#   a. nvidia-smi, start the wall-clock timer
#   b. install train/requirements-train.txt into .venv-train and train/requirements-serve.txt into
#      .venv-serve (uv; nemo-automodel and vllm pin different quack-kernels, so they cannot share a venv),
#      then check torch sees the GPU in both
#      (+ preflight: token lengths and the loss mask, on the real tokenizer and ChatDataset)
#   c. serve base Qwen with vLLM, run scripts/run_prompted.py --name qwen_prompted, stop the server
#   d. train the LoRA (time-boxed; on failure it stops and says so, the PEFT fallback is your call)
#   e. serve base + adapter with vLLM --enable-lora, run run_prompted.py --name lora, stop the server
#   f. score both with scripts/score.py, print the table and total wall-clock minutes
#   g. tar results/, logs/ and the adapter (no optimizer state) into one file to scp back
# If a step fails, whatever exists is still tarred (partial) so the logs come back too.
#
# Flags:
#   --skip-baseline   skip step c (rerun just training)
#   --skip-train      skip step d and reuse checkpoints/lora-v1/adapter (rerun just e-g, e.g. after vLLM
#                     rejected the adapter); does not retrain
#   --skip-install    skip the install in step b if both venvs are already set up (still verifies them)
#   --allow-rescore   allow training/scoring again after results/lora_val_scores.json exists.
#                     Val is the held-out test set and is meant to be scored once per model:
#                     only use this for an infrastructure failure, never to chase a better number.
#   --dry-run         see above
# Environment: TRAIN_TIMEOUT_MIN (default 45) is the time box for step d, including model download.
#              PY=<python> picks the interpreter for --dry-run (needs PyYAML).

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SKIP_BASELINE=0
SKIP_INSTALL=0
SKIP_TRAIN=0
ALLOW_RESCORE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-baseline) SKIP_BASELINE=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --allow-rescore) ALLOW_RESCORE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

TRAIN_TIMEOUT_MIN="${TRAIN_TIMEOUT_MIN:-45}"
# Two venvs: NeMo AutoModel (training, preflight) and vLLM (serving, run_prompted, score).
TRAIN_VENV="$REPO/.venv-train"
TRAIN_PY="$TRAIN_VENV/bin/python"
SERVE_VENV="$REPO/.venv-serve"
SERVE_PY="$SERVE_VENV/bin/python"
PORT=8000
BASE_URL="http://localhost:$PORT/v1"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONFIG="train/lora_qwen25_1p5b.yaml"
CKPT_DIR="checkpoints/lora-v1"
ADAPTER_DIR="$CKPT_DIR/adapter"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
TARBALL="$REPO/gpu_session_$STAMP.tar.gz"

# The commands the real run uses; --dry-run prints these too, so the plan cannot drift.
VLLM_BASE_ARGS=("$BASE_MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 1)
VLLM_LORA_ARGS=("$BASE_MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 1
                --enable-lora --max-lora-rank 16 --lora-modules "lora=$REPO/$ADAPTER_DIR")
TRAIN_CMD=("$TRAIN_VENV/bin/torchrun" --standalone --nproc-per-node 1 -m nemo_automodel.cli.app "$CONFIG")
POD_TOOLS="nvidia-smi python3 curl tar timeout setsid awk grep"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep the model download on the persistent volume if there is one.
if [ -d /workspace ]; then export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"; fi

SECONDS=0
CURRENT_STEP="startup"
SERVER_PID=""

say() { printf '\n=== [%s UTC | %s min] %s\n' "$(date -u +%H:%M:%S)" "$(minutes)" "$*"; }
minutes() { awk -v s="$SECONDS" 'BEGIN { printf "%.1f", s / 60 }'; }
quoted() { printf '%q ' "$@"; echo; }

make_tarball() {
  local items=() p
  for p in results logs "$ADAPTER_DIR" "$CKPT_DIR/training.jsonl" "$CKPT_DIR/validation.jsonl"; do
    if [ -e "$p" ]; then items+=("$p"); fi
  done
  if [ "${#items[@]}" -eq 0 ]; then echo "nothing to tar"; return 1; fi
  tar -czf "$TARBALL" "${items[@]}"
  echo "tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
  echo "contents: ${items[*]}"
}

stop_server() {
  if [ -z "${SERVER_PID:-}" ]; then return 0; fi
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

fail() {
  trap - ERR INT TERM
  set +e
  echo
  echo "!!! STOPPED in step: $CURRENT_STEP"
  echo "!!! $*"
  echo "!!! Wall-clock so far: $(minutes) min. The pod is still billing: scp the tarball back, then stop it."
  stop_server
  make_tarball
  echo "!!! scp it back with:  scp -P <port> root@<pod-ip>:$TARBALL ."
  exit 1
}
trap 'fail "command failed (exit $?): $BASH_COMMAND"' ERR
trap 'fail "interrupted"' INT TERM

# run_step <log name> <command...>: output goes to the screen and logs/<log name>.log.
# Returns the command's own exit status (pipefail), so callers can use `if ! run_step ...`.
run_step() {
  local name="$1"; shift
  # trap - ERR: with set -E the left side of the pipe (a subshell) inherits the ERR trap, so a failure
  # used to run fail() twice: once in the subshell and once for the pipeline in the main shell.
  { trap - ERR; echo "+ $*"; "$@"; } 2>&1 | tee "logs/$name.log"
}

wait_gpu_free() {
  # vLLM holds most of the VRAM until its processes are gone; training needs it back.
  local i used
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [ "${used:-99999}" -lt 1500 ]; then return 0; fi
    sleep 2
  done
  fail "GPU memory is still in use after stopping vLLM (${used} MiB used). Check nvidia-smi."
}

start_server() { # start_server <log name> <vllm serve args...>
  local name="$1"; shift
  say "starting vLLM: vllm serve $*"
  setsid "$SERVE_VENV/bin/vllm" serve "$@" --port "$PORT" > "logs/$name.log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 180); do  # up to 15 min: first run downloads the model and captures CUDA graphs
    if curl -sf "$BASE_URL/models" > /dev/null 2>&1; then echo "vLLM is up after ~$((i * 5))s"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -40 "logs/$name.log"
      fail "vLLM exited before it was ready (full log: logs/$name.log)"
    fi
    sleep 5
  done
  tail -40 "logs/$name.log"
  fail "vLLM was not ready after 15 minutes (full log: logs/$name.log)"
}

check_preds() { # check_preds <name>: every val row must have a prediction
  local want have
  want="$(grep -c . data/val.jsonl)"
  have="$(grep -c . "results/$1_val_preds.jsonl")"
  if [ "$want" != "$have" ]; then fail "results/$1_val_preds.jsonl has $have rows, data/val.jsonl has $want"; fi
  echo "$1: $have/$want rows predicted"
}

run_prompted() { # run_prompted <name> <model id served by vLLM>
  run_step "run_$1" "$SERVE_PY" scripts/run_prompted.py --backend openai --base-url "$BASE_URL" \
    --model "$2" --name "$1" --data data/val.jsonl
  check_preds "$1"
}

# --------------------------------------------------------------------------- dry run
dry_run() {
  local problems=0 out f CHECK_PY
  CHECK_PY="${PY:-python3}"  # PY=... points the checks at a Python that has PyYAML
  if ! "$CHECK_PY" -c 'pass' 2> /dev/null; then CHECK_PY=python; fi
  ok() { echo "  ok    $*"; }
  bad() { echo "  FAIL  $*"; problems=$((problems + 1)); }
  info() { echo "  info  $*"; }

  say "dry run: no GPU, nothing is installed, started or sent"
  echo "repo: $REPO   python: $CHECK_PY"
  echo "checking paths"
  for f in "$CONFIG" train/requirements-train.txt train/requirements-serve.txt shared/template_v1.txt data/val.jsonl \
           data/nemo/train.jsonl data/nemo/val.jsonl scripts/common.py scripts/run_prompted.py \
           scripts/score.py scripts/preflight_train.py; do
    if [ -f "$f" ]; then ok "$f"; else bad "missing: $f"; fi
  done
  for f in $POD_TOOLS uv; do
    if command -v "$f" > /dev/null; then info "$f is here"; else info "$f not found here (step a checks the pod tools; uv is installed in step b)"; fi
  done
  if bash -n "${BASH_SOURCE[0]}"; then ok "this script parses (bash -n)"; else bad "bash -n failed on this script"; fi

  echo "checking the YAML and the data"
  if out="$("$CHECK_PY" scripts/preflight_train.py 2>&1)"; then ok "preflight_train.py"; else bad "preflight_train.py:"; fi
  echo "$out" | sed 's/^/        /'
  for f in data/val.jsonl data/nemo/train.jsonl data/nemo/val.jsonl; do
    info "$f: $(grep -c . "$f") rows"
  done

  echo "checking run_prompted.py builds both requests (sends nothing)"
  for f in "qwen_prompted|$BASE_MODEL" "lora|lora"; do
    if out="$("$CHECK_PY" scripts/run_prompted.py --backend openai --base-url "$BASE_URL" --model "${f#*|}" \
              --name "${f%%|*}" --data data/val.jsonl --dry-run 2>&1)"; then
      ok "run_prompted.py --name ${f%%|*} --model ${f#*|}: $(echo "$out" | head -1)"
    else
      bad "run_prompted.py --name ${f%%|*}: $out"
    fi
  done
  if "$CHECK_PY" scripts/score.py --help > /dev/null 2>&1; then ok "score.py loads"; else bad "score.py --help failed"; fi

  echo
  echo "plan (what a real run executes, in order):"
  echo "  a. nvidia-smi; timer starts"
  echo "  b. uv venv $TRAIN_VENV; uv pip install -r train/requirements-train.txt"
  echo "     uv venv $SERVE_VENV; uv pip install -r train/requirements-serve.txt; check torch.cuda in both; then"
  echo "     $TRAIN_PY scripts/preflight_train.py --tokenizer"
  if [ "$SKIP_BASELINE" = 1 ]; then
    echo "  c. SKIPPED (--skip-baseline)"
  else
    echo "  c. vllm serve $(quoted "${VLLM_BASE_ARGS[@]}" --port "$PORT")"
    echo "     $SERVE_PY scripts/run_prompted.py --backend openai --base-url $BASE_URL --model $BASE_MODEL --name qwen_prompted --data data/val.jsonl"
    echo "     stop the server, wait for the GPU to free"
  fi
  if [ "$SKIP_TRAIN" = 1 ]; then
    echo "  d. SKIPPED (--skip-train): reuse $ADAPTER_DIR"
  else
    echo "  d. timeout ${TRAIN_TIMEOUT_MIN}m $(quoted "${TRAIN_CMD[@]}")"
    echo "     then copy the last epoch's model/ to $ADAPTER_DIR"
  fi
  echo "  e. vllm serve $(quoted "${VLLM_LORA_ARGS[@]}" --port "$PORT")"
  echo "     $SERVE_PY scripts/run_prompted.py --backend openai --base-url $BASE_URL --model lora --name lora --data data/val.jsonl"
  echo "  f. score.py for qwen_prompted and lora; print results/summary.md and total minutes"
  echo "  g. tar results/ logs/ $ADAPTER_DIR (+ training.jsonl, validation.jsonl) -> gpu_session_<stamp>.tar.gz"
  echo
  if [ "$problems" -eq 0 ]; then echo "dry run: all checks passed"; else echo "dry run: $problems check(s) FAILED"; fi
  [ "$problems" -eq 0 ]
}

if [ "$DRY_RUN" = 1 ]; then
  trap - ERR INT TERM
  dry_run
  exit $?
fi

# --------------------------------------------------------------------------- a
CURRENT_STEP="a: environment"
mkdir -p logs results
say "a. environment"
for t in $POD_TOOLS; do
  if ! command -v "$t" > /dev/null; then fail "'$t' is not installed on this machine (the script needs it later)"; fi
done
run_step 01_env bash -c 'nvidia-smi; echo; (git rev-parse HEAD; git status --short | head) 2>&1 || echo "(not a git checkout)"; df -h . | tail -1'
if [ -f "results/lora_val_scores.json" ] && [ "$ALLOW_RESCORE" = 0 ]; then
  fail "results/lora_val_scores.json exists: the LoRA was already scored on the held-out set. Not retraining. Pass --allow-rescore only for an infrastructure failure."
fi

# --------------------------------------------------------------------------- b
CURRENT_STEP="b: install"
say "b. install"
if [ "$SKIP_INSTALL" = 0 ]; then
  if ! command -v uv > /dev/null; then run_step 02_install_uv python3 -m pip install -q uv; fi
  CURRENT_STEP="b: install (train venv)"
  if [ ! -x "$TRAIN_PY" ]; then run_step 02_venv_train uv venv --python ">=3.11" "$TRAIN_VENV"; fi
  run_step 02_install_train uv pip install --python "$TRAIN_PY" -r train/requirements-train.txt
  uv pip freeze --python "$TRAIN_PY" > logs/pip_freeze_train.txt
  CURRENT_STEP="b: install (serve venv)"
  if [ ! -x "$SERVE_PY" ]; then run_step 02_venv_serve uv venv --python ">=3.11" "$SERVE_VENV"; fi
  run_step 02_install_serve uv pip install --python "$SERVE_PY" -r train/requirements-serve.txt
  uv pip freeze --python "$SERVE_PY" > logs/pip_freeze_serve.txt
fi
CURRENT_STEP="b: verify install"
VERIFY_PY='
import importlib, sys
mods = sys.argv[1:]
import torch
for m in mods:
    importlib.import_module(m)
vers = " | ".join(f"{m} {getattr(sys.modules[m], '"'"'__version__'"'"', '"'"'?'"'"')}" for m in ["torch", *mods])
print("python", sys.version.split()[0], "|", vers)
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU. Usual cause: the pod NVIDIA driver is older than this torch build CUDA. "
             "Pick a pod template with a newer driver (or CUDA 12.8+) and start over.")
print("gpu:", torch.cuda.get_device_name(0))
'
if ! run_step 02_verify_train "$TRAIN_PY" -c "$VERIFY_PY" transformers nemo_automodel yaml; then
  fail "train venv check failed (see logs/02_verify_train.log). Nothing has been trained."
fi
if ! run_step 02_verify_serve "$SERVE_PY" -c "$VERIFY_PY" transformers vllm openai; then
  fail "serve venv check failed (see logs/02_verify_serve.log). Nothing has been trained."
fi
CURRENT_STEP="b: preflight (token lengths, loss mask)"
if ! run_step 03_preflight "$TRAIN_PY" scripts/preflight_train.py --tokenizer; then
  fail "preflight failed (logs/03_preflight.log). If it is the loss mask or a too-long row, stop here: training would be wrong."
fi

# --------------------------------------------------------------------------- c
if [ "$SKIP_BASELINE" = 1 ]; then
  say "c. skipped (--skip-baseline)"
else
  CURRENT_STEP="c: prompted base Qwen"
  say "c. prompted base Qwen"
  start_server 04_vllm_base "${VLLM_BASE_ARGS[@]}"
  run_prompted qwen_prompted "$BASE_MODEL"
  stop_server
  wait_gpu_free
fi

# --------------------------------------------------------------------------- d
if [ "$SKIP_TRAIN" = 1 ]; then
  say "d. skipped (--skip-train): reusing $ADAPTER_DIR"
  if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ] || [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    fail "--skip-train needs $ADAPTER_DIR/{adapter_model.safetensors,adapter_config.json}, which are not there"
  fi
else
CURRENT_STEP="d: train"
say "d. train (time box: ${TRAIN_TIMEOUT_MIN} min)"
if [ -d "$CKPT_DIR" ]; then
  # A rerun after a failed attempt: don't let old epoch dirs or appended logs mix with the new run.
  mv "$CKPT_DIR" "$CKPT_DIR.prev-$STAMP"
  echo "moved the existing $CKPT_DIR aside to $CKPT_DIR.prev-$STAMP"
fi
if run_step 05_train timeout -k 30 "${TRAIN_TIMEOUT_MIN}m" "${TRAIN_CMD[@]}"; then rc=0; else rc=$?; fi
if [ "$rc" -ne 0 ]; then
  tail -30 logs/05_train.log
  if [ "$rc" -eq 124 ]; then why="TIMED OUT after ${TRAIN_TIMEOUT_MIN} min"; else why="FAILED (exit $rc)"; fi
  fail "TRAINING $why. Full log: logs/05_train.log. Nothing after this step was run. Your call: fix and rerun with --skip-baseline --skip-install, or fall back to PEFT + TRL."
fi

CURRENT_STEP="d: collect the adapter"
last_epoch_dir="$(ls -d "$CKPT_DIR"/epoch_*_step_* 2> /dev/null | sort -V | tail -1 || true)"
if [ -z "$last_epoch_dir" ]; then fail "training exited 0 but $CKPT_DIR has no epoch_*_step_* checkpoint"; fi
echo "checkpoints written: $(cd "$CKPT_DIR" && ls -d epoch_*_step_* | sort -V | tr '\n' ' ')"
if [ ! -f "$last_epoch_dir/model/adapter_model.safetensors" ] || [ ! -f "$last_epoch_dir/model/adapter_config.json" ]; then
  fail "$last_epoch_dir/model has no adapter_model.safetensors + adapter_config.json (it holds: $(ls "$last_epoch_dir/model" | tr '\n' ' '))"
fi
rm -rf "$ADAPTER_DIR"
mkdir -p "$ADAPTER_DIR"
cp "$last_epoch_dir"/model/adapter_model.safetensors "$last_epoch_dir"/model/adapter_config.json "$ADAPTER_DIR"/
echo "final adapter = $(basename "$last_epoch_dir") (the last epoch; LOWEST_VAL is not used)"
"$TRAIN_PY" - "$ADAPTER_DIR/adapter_config.json" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print({k: c.get(k) for k in ("peft_type", "r", "lora_alpha", "target_modules", "base_model_name_or_path")})
assert c.get("r") == 16 and c.get("lora_alpha") == 32, "adapter rank/alpha is not 16/32"
PY
if [ -f "$CKPT_DIR/validation.jsonl" ]; then echo "validation loss per epoch:"; cat "$CKPT_DIR/validation.jsonl"; fi
fi  # end of step d (not skipped)

# --------------------------------------------------------------------------- e
CURRENT_STEP="e: serve base + adapter"
say "e. prompted base + adapter"
if [ -f results/lora_val_preds.jsonl ]; then
  mv results/lora_val_preds.jsonl "results/lora_val_preds.prev-$STAMP.jsonl"
  echo "moved an older results/lora_val_preds.jsonl aside (run_prompted would have resumed from it)"
fi
start_server 06_vllm_lora "${VLLM_LORA_ARGS[@]}"
if ! curl -sf "$BASE_URL/models" | grep -q '"lora"'; then
  fail "vLLM is up but did not register the adapter 'lora' (see logs/06_vllm_lora.log)"
fi
run_prompted lora lora
stop_server

# --------------------------------------------------------------------------- f
CURRENT_STEP="f: score"
say "f. score"
for name in qwen_prompted lora; do
  if [ ! -f "results/${name}_val_preds.jsonl" ]; then echo "$name: no predictions here, not scored"; continue; fi
  if [ -f "results/${name}_val_scores.json" ] && [ "$ALLOW_RESCORE" = 0 ] && [ "$name" != lora ]; then
    echo "$name: already scored, not rescoring (val is scored once; --allow-rescore to override)"; continue
  fi
  run_step "07_score_$name" "$SERVE_PY" scripts/score.py --gold data/val.jsonl \
    --preds "results/${name}_val_preds.jsonl" --name "$name"
done
echo
echo "results/summary.md:"
cat results/summary.md
echo
echo "TOTAL WALL-CLOCK: $(minutes) minutes (from script start; multiply by the pod's hourly rate for the cost)"

# --------------------------------------------------------------------------- g
CURRENT_STEP="g: tarball"
say "g. tarball"
trap - ERR
make_tarball
echo
echo "Next: scp the tarball back, then STOP THE POD (stopping the script does not stop billing)."
