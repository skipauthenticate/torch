#!/usr/bin/env bash
# Installs the engine, the server and the model files on a Jetson AGX Orin (docs/install.md): checks the system,
# builds into a prefix, makes the server's virtual environment, downloads or verifies the two model shards, builds
# the draft-layer file on this machine and runs a smoke test. Never uses sudo. Safe to run again: each step keeps what
# is already done (a download resumes). Prints every path it writes.
set -euo pipefail

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GGUF_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
GGUF_REVISION=ed59f92082b1e93c0e96d60a8b11aab089b52f09
BASE_REPO=Qwen/Qwen3.8-Flash-Next
BASE_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
SHARD1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
SHARD2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
# The smoke test: a one-question chat prompt (thinking off) and its 16 greedy tokens, recorded on the Orin.
SMOKE_PROMPT=248045,846,198,3710,369,279,6511,314,9338,30,21134,303,799,11316,13,248046,198,248045,74455,198,248068,271,248069,271
SMOKE_IDS=760,6511,314,9338,369,11751,13,248046,198,248044,248045,846,198,248046,198,248045

usage() {
    cat <<'EOF'
usage: scripts/install.sh [options]

  --accept-model-licenses  accept the two model repositories' licenses (printed first) without a prompt
  --prefix DIR             where to install (default ~/.local): bin/, lib/, include/ and share/torch/
  --models DIR             where the two model shards are or go (default PREFIX/share/torch/models); shards
                           already there are verified in place, missing ones downloaded, partial ones resumed
  --steps LIST             some of build,server,models,draft,smoke (default all, in that order)
  --jobs N                 parallel compile jobs (default 4)
  --wrap CMD               run the build and the smoke test through CMD (a shared machine's GPU lock)
  --check                  only check the system; change nothing
  --no-gpu-check           skip the Jetson, JetPack, CUDA and memory checks (CI)

HF_TOKEN, when set, is sent to Hugging Face as a bearer token (never printed).
EOF
}

PREFIX="$HOME/.local" MODELS="" ACCEPT=0 CHECK_ONLY=0 GPU_CHECK=1 JOBS=4 STEPS="build,server,models,draft,smoke" WRAP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --accept-model-licenses) ACCEPT=1 ;;
        --prefix) PREFIX=$2; shift ;;
        --models) MODELS=$2; shift ;;
        --steps) STEPS=$2; shift ;;
        --jobs) JOBS=$2; shift ;;
        --wrap) WRAP=$2; shift ;;
        --check) CHECK_ONLY=1 ;;
        --no-gpu-check) GPU_CHECK=0 ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done
mkdir_p() { [ -d "$1" ] || { mkdir -p "$1"; echo "wrote $1/"; }; }
die() { echo "install.sh: $*" >&2; exit 1; }
step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
wrapped() {
    if [ -n "$WRAP" ]; then
        local -a w
        read -ra w <<<"$WRAP"
        "${w[@]}" "$@"
    else
        "$@"
    fi
}
sha_of() { awk -v f="$1" '$2 == f { print $1 }' "$SRC/scripts/models.sha256"; }
free_bytes() { df -P -B1 "$1" | awk 'NR == 2 { print $4 }'; }
need_space() {  # need_space DIR BYTES WHAT
    local free
    free=$(free_bytes "$1")
    [ "$free" -ge "$2" ] || die "$3 needs $(($2 / 1000000000)) GB free in $1, which has $((free / 1000000000)) GB."
}

# ---- checks ---------------------------------------------------------------------------------------------------------
NVCC="${CUDACXX:-}"
[ -n "$NVCC" ] || NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
check_system() {
    local failed=0 apt=() version
    ok() { printf '  ok       %s\n' "$1"; }
    missing() { printf '  MISSING  %s\n' "$1"; failed=1; [ -z "${2:-}" ] || apt+=("$2"); }
    echo "checking the system"
    if [ "$GPU_CHECK" = 1 ]; then
        version=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)
        case "$version" in 7.*) ok "JetPack $version" ;; *) missing "JetPack 7 (nvidia-jetpack: ${version:-not installed})" ;; esac
        if tr '\0' ' ' </proc/device-tree/compatible 2>/dev/null | grep -q 'nvidia,tegra234'; then
            ok "Jetson AGX Orin (tegra234, sm_87)"
        else
            missing "a Jetson AGX Orin (tegra234, sm_87)"
        fi
        version=$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' || true)
        case "$version" in 13.*) ok "CUDA $version ($NVCC)" ;; *) missing "CUDA 13.x nvcc (set CUDACXX; found: ${version:-none})" ;; esac
        version=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
        if [ "$version" -ge 60 ]; then ok "memory ${version} GiB"; else missing "64 GB of memory (found ${version} GiB)"; fi
    fi
    version=$(cmake --version 2>/dev/null | sed -n 's/cmake version \([0-9.]*\).*/\1/p' || true)
    if [ -n "$version" ] && printf '3.24\n%s\n' "$version" | sort -C -V; then ok "cmake $version"; else missing "cmake >= 3.24 (found ${version:-none})" cmake; fi
    if command -v ninja >/dev/null; then ok "ninja $(ninja --version)"; else missing "ninja" ninja-build; fi
    if printf '#include <span>\nint main() { std::span<int> s; return static_cast<int>(s.size()); }\n' |
        "${CXX:-c++}" -std=c++20 -x c++ - -o /dev/null 2>/dev/null; then
        ok "C++20 compiler (${CXX:-c++})"
    else
        missing "a C++20 compiler" g++
    fi
    version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    if [ "$version" = 3.12 ] && python3 -c 'import venv, ensurepip' 2>/dev/null; then
        ok "python 3.12 with venv"
    else
        missing "python 3.12 with venv (found ${version:-none})" python3-venv
    fi
    for tool in curl sha256sum; do
        if command -v "$tool" >/dev/null; then ok "$tool"; else missing "$tool" "$([ "$tool" = curl ] && echo curl || echo coreutils)"; fi
    done
    if [ ${#apt[@]} -gt 0 ]; then
        echo "install the missing packages with: sudo apt-get install -y ${apt[*]}"
    fi
    return $failed
}

if [ "$CHECK_ONLY" = 1 ]; then
    check_system
    exit
fi
check_system || die "fix the checks above and run again."

# ---- licenses -------------------------------------------------------------------------------------------------------
cat <<EOF

This project's code is MIT. The model files are not part of it: the installer downloads them, and builds the draft
layer from another model, under the repositories' own licenses:
  $GGUF_REPO (revision ${GGUF_REVISION:0:8}): Apache-2.0
    https://huggingface.co/$GGUF_REPO
  $BASE_REPO (revision ${BASE_REVISION:0:8}): "other" on its model card (qwen-community-1.0); the
    quantized model is derived from it, and the draft layer is built from it on this machine
    https://huggingface.co/$BASE_REPO/blob/$BASE_REVISION/LICENSE

EOF
if [ "$ACCEPT" != 1 ]; then
    if [ -t 0 ]; then
        read -r -p "Type yes to accept both licenses: " answer
        [ "$answer" = yes ] || die "the licenses were not accepted; nothing was downloaded."
    else
        die "read the licenses above and pass --accept-model-licenses to accept them."
    fi
fi

SHARE="$PREFIX/share/torch"
MODELS="${MODELS:-$SHARE/models}"
mkdir_p "$SHARE"

# ---- build ----------------------------------------------------------------------------------------------------------
if step build; then
    echo "building into $SHARE/build"
    nice -n 10 cmake -S "$SRC" -B "$SHARE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" >/dev/null
    wrapped nice -n 10 cmake --build "$SHARE/build" --target torch torch_engine_shared -j "$JOBS"
    for f in "bin/torch:$SHARE/build/torch:755" "lib/libtorch_engine.so:$SHARE/build/libtorch_engine.so:644" \
        "include/torch_engine.h:$SRC/include/torch_engine.h:644"; do
        IFS=: read -r to from mode <<<"$f"
        install -D -m "$mode" "$from" "$PREFIX/$to"
        echo "wrote $PREFIX/$to"
    done
fi

# ---- server ---------------------------------------------------------------------------------------------------------
if step server; then
    if [ ! -x "$SHARE/venv/bin/python" ]; then
        python3 -m venv "$SHARE/venv"
        echo "wrote $SHARE/venv/"
    fi
    "$SHARE/venv/bin/pip" install --quiet --disable-pip-version-check --no-deps --require-hashes -r "$SRC/server/requirements.txt"
    rm -rf "$SHARE/server"
    mkdir -p "$SHARE/server"
    cp -R "$SRC/server/torch_server" "$SHARE/server/"
    find "$SHARE/server" -name __pycache__ -prune -exec rm -rf {} +
    echo "wrote $SHARE/server/torch_server/"
    rm -rf "$SHARE/calibration"  # the placement and draft vocabulary the server uses by default
    cp -R "$SRC/calibration" "$SHARE/calibration"
    echo "wrote $SHARE/calibration/"
    cat >"$PREFIX/bin/torch-server" <<EOF
#!/bin/sh
# torch_server with the files scripts/install.sh installed (docs/server.md); more options after these are passed on.
PYTHONPATH="$SHARE/server" exec "$SHARE/venv/bin/python" -m torch_server --library "$PREFIX/lib/libtorch_engine.so" \\
  --torch-bin "$PREFIX/bin/torch" --model "$MODELS/$SHARD1" --draft "$SHARE/draft/draft-iq4.gguf" "\$@"
EOF
    chmod 755 "$PREFIX/bin/torch-server"
    echo "wrote $PREFIX/bin/torch-server"
fi

# ---- models ---------------------------------------------------------------------------------------------------------
# A file is hashed once: the stamp records its sha256, size and time, and a later run skips a file that still matches.
verified() {  # verified PATH NAME: true when PATH has NAME's sha256 (hashing it unless its stamp still matches)
    local want got stamp="$SHARE/verified/$2" now
    want=$(sha_of "$2")
    now="$want $(stat -L -c '%s %Y' "$1")"
    [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$now" ] && return 0
    echo "verifying $1"
    got=$(sha256sum "$1" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { echo "  sha256 $got, expected $want" >&2; return 1; }
    mkdir -p "$SHARE/verified"
    echo "$now" >"$stamp"
    echo "wrote $stamp"
}
curl_auth() {  # the token reaches curl through a file descriptor, never the command line
    if [ -n "${HF_TOKEN:-}" ]; then
        curl --config <(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN") "$@"
    else
        curl "$@"
    fi
}
if step models; then
    mkdir_p "$MODELS"
    for f in "$SHARD1" "$SHARD2"; do
        path="$MODELS/$f"
        if [ -e "$path" ]; then
            verified "$path" "$f" || die "$path is not the expected file: move it away and run again to download it."
            continue
        fi
        url="https://huggingface.co/$GGUF_REPO/resolve/$GGUF_REVISION/IQ3_S/$f"
        size=$(curl_auth --silent --fail --location --head "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }')
        [ -n "$size" ] || die "could not reach $url (set HF_TOKEN if the repository needs one)."
        have=$([ -f "$path.part" ] && stat -c %s "$path.part" || echo 0)
        need_space "$MODELS" $((size - have)) "$f"
        echo "downloading $url to $path.part ($((size / 1000000000)) GB, resuming from $((have / 1000000)) MB)"
        curl_auth --fail --location --continue-at - --retry 5 --retry-delay 10 --progress-bar --output "$path.part" "$url"
        echo "wrote $path.part"
        verified "$path.part" "$f" || die "$path.part does not have the expected sha256: delete it and run again."
        mv "$path.part" "$path"  # same size and time: the stamp still matches
        echo "wrote $path"
    done
fi

# ---- draft layer ----------------------------------------------------------------------------------------------------
if step draft; then
    draft="$SHARE/draft/draft-iq4.gguf"
    if [ -f "$draft" ] && [ -f "$SHARE/verified/draft-iq4.gguf" ]; then
        echo "draft layer: $draft is built"
    else
        mkdir_p "$SHARE/draft"
        if [ ! -x "$SHARE/draft-venv/bin/python" ]; then
            python3 -m venv "$SHARE/draft-venv"
            echo "wrote $SHARE/draft-venv/"
        fi
        "$SHARE/draft-venv/bin/pip" install --quiet --disable-pip-version-check --no-deps --require-hashes \
            -r "$SRC/scripts/draft-requirements.txt"
        need_space "$SHARE/draft" 7000000000 "the draft layer (a 5.2 GB intermediate and the 1.48 GB file)"
        echo "building the draft layer from $BASE_REPO's 31 mtp tensors (range requests, about 5.2 GB)"
        "$SHARE/draft-venv/bin/python" "$SRC/tools/make_draft.py" "$SHARE/draft" --experts iq4 --bf16-only
        echo "wrote $SHARE/draft/draft-bf16.gguf"
        verified "$SHARE/draft/draft-bf16.gguf" draft-bf16.gguf ||
            die "the BF16 tensors from $BASE_REPO differ from the recorded ones: delete $SHARE/draft and run again."
        nice -n 10 "$SHARE/draft-venv/bin/python" "$SRC/tools/make_draft.py" "$SHARE/draft" --experts iq4
        echo "wrote $draft"
        if ! verified "$draft" draft-iq4.gguf; then  # the draft changes speed, never output
            echo "install.sh: warning: $draft's sha256 differs from the recorded build's; the BF16 input matched." >&2
            mkdir -p "$SHARE/verified"
            echo "unrecorded $(stat -c '%s %Y' "$draft")" >"$SHARE/verified/draft-iq4.gguf"
        fi
        rm -f "$SHARE/draft/draft-bf16.gguf" "$SHARE/draft/draft-bf16.done"
        echo "removed the BF16 intermediate (5.2 GB)"
    fi
fi

# ---- smoke test -----------------------------------------------------------------------------------------------------
if step smoke; then
    echo "smoke test: 16 greedy tokens"
    wrapped "$PREFIX/bin/torch" run "$MODELS/$SHARD1" --tokens "$SMOKE_PROMPT" -n 16 --quiet >"$SHARE/smoke.log" 2>&1 ||
        die "the smoke test failed: see $SHARE/smoke.log"
    echo "wrote $SHARE/smoke.log"
    got=$(sed -n 's/^generated tokens: //p' "$SHARE/smoke.log")
    [ "$got" = "$SMOKE_IDS" ] || die "the smoke test generated $got, expected $SMOKE_IDS ($SHARE/smoke.log)."
    echo "smoke test: the expected tokens"
fi

cat <<EOF

installed. Start the server with:
  $PREFIX/bin/torch-server --context 16384
and talk to it at http://127.0.0.1:8080/v1 (docs/server.md).
EOF
