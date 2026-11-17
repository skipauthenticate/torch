#!/bin/sh
# The checks a milestone passes on the Orin before it merges (CONTRIBUTING.md). Builds the tree, then on the GPU runs
#   - the unit and kernel parity tests, and the model parity test against tests/model/reference.json;
#   - the decode identity checks: byte-identical logits from the CUDA graph and kernel by kernel, and with 0, 256 and
#     the budget's experts resident (output must not depend on residency); --fast-prefill deterministic and the same
#     at three chunk sizes;
#   - the kernel harness against the decode gates (bench/kernel_gates.py).
# It prints each result and exits non-zero if any check fails.
#
# usage, on the Orin:          TORCH_MODEL=/path/MODEL-00001-of-0000N.gguf tools/check_orin.sh
#        from another machine: TORCH_ORIN=user@host TORCH_MODEL=/path/on/board/... tools/check_orin.sh
#   TORCH_ORIN       copy this tree to the board first (rsync), into TORCH_ORIN_DIR under its home folder
#                    (default torch-check), and run the checks there
#   TORCH_GPU_LOCK   a lock file the board's GPU users share; when set, the GPU checks run under flock on it, and are
#                    stopped after TORCH_GPU_MINUTES (default 19) so a hung run cannot keep the lock
#   TORCH_CUDA_BIN   the CUDA toolkit's bin folder (default /usr/local/cuda/bin)
#   TORCH_REF        the model parity reference for TORCH_MODEL (default tests/model/reference.json, the IQ3_S file's);
#                    TORCH_TIE and TORCH_MAX_KL set its thresholds (default parity.py's, set from that file's spread)
#   TORCH_KERNEL_GATES=report  print the kernel gates without failing on them (a model file whose kernels are not
#                    tuned yet)
set -eu
: "${TORCH_MODEL:?set TORCH_MODEL to the path of shard 1 (on the board)}"
root=$(cd "$(dirname "$0")/.." && pwd)

if [ -n "${TORCH_ORIN:-}" ]; then
    dir=${TORCH_ORIN_DIR:-torch-check}
    sha=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
    # --checksum, not timestamps: a tree switched between branches can keep old modification times.
    rsync -rlpD --checksum --delete --exclude build --exclude .git --exclude .venv "$root/" "$TORCH_ORIN:$dir/"
    # -n: no stdin; with it the session could stay open after the checks ended.
    exec ssh -n "$TORCH_ORIN" "cd \"\$HOME/$dir\" && TORCH_MODEL='$TORCH_MODEL' TORCH_GIT_SHA='$sha' \
        TORCH_GPU_LOCK='${TORCH_GPU_LOCK:-}' TORCH_GPU_MINUTES='${TORCH_GPU_MINUTES:-19}' \
        TORCH_CUDA_BIN='${TORCH_CUDA_BIN:-}' TORCH_REF='${TORCH_REF:-}' TORCH_TIE='${TORCH_TIE:-}' \
        TORCH_MAX_KL='${TORCH_MAX_KL:-}' TORCH_KERNEL_GATES='${TORCH_KERNEL_GATES:-}' sh tools/check_orin.sh"
fi

cd "$root"
out=build/check

if [ "${TORCH_CHECK_STAGE:-}" != gpu ]; then
    export PATH="${TORCH_CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
    sha=${TORCH_GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}
    mkdir -p build
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTORCH_WERROR=ON -DTORCH_GIT_SHA="$sha" > /dev/null
    nice -n 10 cmake --build build -j 4 > build/check-build.log 2>&1 || { tail -40 build/check-build.log; exit 1; }
    echo "build ok ($sha)"
    if [ -n "${TORCH_GPU_LOCK:-}" ]; then
        echo "waiting for the GPU lock $TORCH_GPU_LOCK"
        # timeout signals its whole process group, so no child keeps the lock's descriptor open past the limit.
        exec flock -w 3600 "$TORCH_GPU_LOCK" timeout -k 10 "$((${TORCH_GPU_MINUTES:-19} * 60))" \
            env TORCH_CHECK_STAGE=gpu sh "$root/tools/check_orin.sh"
    fi
fi

failed=0
mkdir -p "$out/model"
run() { ./build/torch run "$TORCH_MODEL" --tokens 760,6511,314,9338,369 -n 4 "$@" > /dev/null; }

if ctest --test-dir build -L unit --output-on-failure > "$out/unit.log" 2>&1; then echo "ok   unit tests"; else
    echo "FAIL unit tests"; tail -30 "$out/unit.log"; failed=1; fi
if ctest --test-dir build -L parity --output-on-failure > "$out/parity.log" 2>&1; then echo "ok   kernel parity tests"
else echo "FAIL kernel parity tests"; tail -30 "$out/parity.log"; failed=1; fi
thresholds=""
[ -n "${TORCH_TIE:-}" ] && thresholds="--tie $TORCH_TIE"
[ -n "${TORCH_MAX_KL:-}" ] && thresholds="$thresholds --max-mean-kl $TORCH_MAX_KL"
# shellcheck disable=SC2086 # thresholds is a list of options
if python3 tests/model/parity.py compare --torch build/torch --model "$TORCH_MODEL" \
    --ref "${TORCH_REF:-tests/model/reference.json}" $thresholds --work "$out/model" > "$out/model-parity.log" 2>&1
then echo "ok   model parity: $(tail -1 "$out/model-parity.log")"
else echo "FAIL model parity"; tail -12 "$out/model-parity.log"; failed=1; fi

if run --logits "$out/graph.bin" && TORCH_EAGER=1 run --logits "$out/eager.bin" &&
    run --resident 0 --logits "$out/r0.bin" && run --resident 256 --logits "$out/r256.bin"; then
    if cmp -s "$out/graph.bin" "$out/eager.bin"; then echo "ok   graph = eager"; else
        echo "FAIL graph and eager logits differ"; failed=1; fi
    if cmp -s "$out/graph.bin" "$out/r0.bin" && cmp -s "$out/graph.bin" "$out/r256.bin"; then
        echo "ok   residency 0, 256 and the budget's give identical logits"
    else echo "FAIL logits depend on residency"; failed=1; fi
else echo "FAIL a decode run"; failed=1; fi

# --fast-prefill (prefill's BF16 products on the tensor cores) is not bitwise the windows, but must be deterministic and
# independent of the chunk size: 1,100 prompt tokens at chunks of 1,100 (twice), 550 and 275 give the same last prompt
# row and the same generated rows (the files' last 4 rows of 248,320 logits).
fast() { P=$(cut -d, -f1-1100 bench/workloads/review-calib-101.tokens)
    ./build/torch run "$TORCH_MODEL" --tokens "$P" -n 4 --fast-prefill --prefill "$1" --logits "$2" > /dev/null; }
if fast 1100 "$out/f1100.bin" && fast 1100 "$out/f1100b.bin" && fast 550 "$out/f550.bin" && fast 275 "$out/f275.bin"
then
    if cmp -s "$out/f1100.bin" "$out/f1100b.bin" && python3 - "$out" <<'PY'
import sys
row = 248320 * 4
tails = [open(f"{sys.argv[1]}/f{c}.bin", "rb").read()[-4 * row:] for c in (1100, 550, 275)]
sys.exit(0 if tails[0] == tails[1] == tails[2] else 1)
PY
    then echo "ok   --fast-prefill deterministic and the same at chunks of 1,100, 550 and 275"
    else echo "FAIL --fast-prefill changes between runs or chunk sizes"; failed=1; fi
else echo "FAIL a --fast-prefill run"; failed=1; fi

if ./build/torch_bench_kernels "$TORCH_MODEL" --json "$out/kernels.json" > "$out/kernels.txt" 2>&1; then
    if [ "${TORCH_KERNEL_GATES:-}" = report ]; then python3 bench/kernel_gates.py "$out/kernels.json" || true
    else python3 bench/kernel_gates.py "$out/kernels.json" || failed=1; fi
else echo "FAIL kernel harness"; tail -20 "$out/kernels.txt"; failed=1; fi
exit $failed
