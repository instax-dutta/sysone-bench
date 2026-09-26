#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Every path is derived from this script's own location or from the environment, so one
# checkout runs for any user on any host with no personal path baked in.
#   SYSONE_BENCH_WORKSPACE_ROOT        override the detected checkout
#   SYSONE_BENCH_MODEL_CACHE           override the model cache (default: $XDG_CACHE_HOME or ~/.cache)
#   SYSONE_BENCH_IMAGE                 override the container image tag
#   SYSONE_BENCH_MANIFEST              override the sealed manifest path
#   SYSONE_BENCH_MANIFEST_CHECKSUM     override the sealed manifest checksum path
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SYSONE_BENCH_WORKSPACE_ROOT:-$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd)}"
RUNS_ROOT="$WORKSPACE_ROOT/runs"
MODEL_CACHE="${SYSONE_BENCH_MODEL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/sysone-bench-v2}"
MANIFEST_PATH="${SYSONE_BENCH_MANIFEST:-$WORKSPACE_ROOT/datasets/v2/manifest.jsonl}"
MANIFEST_CHECKSUM_PATH="${SYSONE_BENCH_MANIFEST_CHECKSUM:-$WORKSPACE_ROOT/datasets/v2/manifest.sha256}"
PROJECT_IMAGE="${SYSONE_BENCH_IMAGE:-sysone-bench-v2:cpu}"
PREFLIGHT_PYTHON="/usr/bin/python3"
VENV_PYTHON="/workspace/.venv/bin/python"
V2_MODULE="benchmark.orchestrator"
HOST_UID="$(/usr/bin/id -u)"
HOST_GID="$(/usr/bin/id -g)"

usage() {
    printf '%s\n' "usage: run_open_model.sh --run-id <id> --model <laya|qwen|router>" >&2
    exit 2
}

run_id=""
model=""
while (($# > 0)); do
    case "$1" in
        --run-id)
            (($# >= 2)) || usage
            run_id="$2"
            shift 2
            ;;
        --model)
            (($# >= 2)) || usage
            model="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || usage
[[ "$run_id" != "." && "$run_id" != ".." ]] || usage
case "$model" in
    laya|qwen|router) ;;
    *) usage ;;
esac
if [[ "$HOST_UID" == "0" ]]; then
    printf '%s\n' "refusing to launch as root" >&2
    exit 1
fi
[[ "$HOST_UID" =~ ^[1-9][0-9]*$ && "$HOST_GID" =~ ^[0-9]+$ ]] || exit 1
container_name="sysone-bench-$run_id"
run_dir="$RUNS_ROOT/$run_id"
OWNER_MARKER_NAME=".sysone-owner"
OWNER_MARKER="$run_dir/$OWNER_MARKER_NAME"
CONTAINER_CIDFILE_NAME=".sysone-cid"
CONTAINER_CIDFILE="$run_dir/$CONTAINER_CIDFILE_NAME"
run_dir_created=0
postflight_recorded=0
container_id=""

validate_json() {
    "$PREFLIGHT_PYTHON" -c 'import json, sys; value = json.load(sys.stdin); raise SystemExit(0 if isinstance(value, dict) else 1)'
}

validate_fixed_paths() {
    [[ -d "$WORKSPACE_ROOT" && ! -L "$WORKSPACE_ROOT" && -O "$WORKSPACE_ROOT" ]] || return 1
    [[ -d "$RUNS_ROOT" && ! -L "$RUNS_ROOT" && -O "$RUNS_ROOT" ]] || return 1
    [[ "$run_dir" == "$RUNS_ROOT/$run_id" ]] || return 1
    [[ -d "$run_dir" && ! -L "$run_dir" && -O "$run_dir" ]] || return 1
}

validate_owned_run() {
    validate_fixed_paths || return 1
    [[ -f "$OWNER_MARKER" && ! -L "$OWNER_MARKER" && -O "$OWNER_MARKER" ]] || return 1
    local marker_contents=""
    marker_contents="$(<"$OWNER_MARKER")" || return 1
    [[ "$marker_contents" == "$run_id" ]] || return 1
}

read_container_id() {
    validate_owned_run || return 1
    [[ -f "$CONTAINER_CIDFILE" && ! -L "$CONTAINER_CIDFILE" && -O "$CONTAINER_CIDFILE" ]] || return 1
    local recorded_id=""
    recorded_id="$(<"$CONTAINER_CIDFILE")" || return 1
    [[ "$recorded_id" =~ ^[0-9a-f]{12,64}$ ]] || return 1
    printf '%s\n' "$recorded_id"
}

create_owner_marker() {
    [[ ! -e "$OWNER_MARKER" && ! -L "$OWNER_MARKER" ]] || return 1
    (
        set -C
        printf '%s\n' "$run_id" > "$OWNER_MARKER"
    ) || return 1
    chmod 600 "$OWNER_MARKER" || return 1
    validate_owned_run
}

record_postflight() {
    if ((postflight_recorded)); then
        return 0
    fi
    local snapshot=""
    local status=0
    if ! validate_owned_run; then
        return 1
    fi
    if snapshot="$("$PREFLIGHT_PYTHON" "$SCRIPT_DIR/preflight.py" --run-id "$run_id" --workspace-root "$WORKSPACE_ROOT" --run-root "$RUNS_ROOT" --model-cache "$MODEL_CACHE" --manifest-path "$MANIFEST_PATH" --manifest-checksum-path "$MANIFEST_CHECKSUM_PATH" --post-run)"; then
        status=0
    else
        status=$?
    fi
    if ! validate_owned_run; then
        return 1
    fi
    if [[ -z "$snapshot" ]] || ! printf '%s' "$snapshot" | validate_json; then
        return 1
    fi
    if ! (
        set -C
        printf '%s\n' "$snapshot" > "$run_dir/postflight.json"
    ); then
        return 1
    fi
    postflight_recorded=1
    return "$status"
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((run_dir_created)); then
        record_postflight || true
        if [[ -z "$container_id" ]]; then
            container_id="$(read_container_id)" || container_id=""
        fi
        if [[ -n "$container_id" ]]; then
            "$SCRIPT_DIR/cleanup.sh" --run-id "$run_id" --container-id "$container_id" || true
        fi
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if ! preflight_json="$("$PREFLIGHT_PYTHON" "$SCRIPT_DIR/preflight.py" --run-id "$run_id" --workspace-root "$WORKSPACE_ROOT" --run-root "$RUNS_ROOT" --model-cache "$MODEL_CACHE" --manifest-path "$MANIFEST_PATH" --manifest-checksum-path "$MANIFEST_CHECKSUM_PATH")" || ! printf '%s' "$preflight_json" | validate_json; then
    exit 1
fi
[[ -f "$MANIFEST_PATH" && ! -L "$MANIFEST_PATH" ]] || exit 1
[[ -f "$MANIFEST_CHECKSUM_PATH" && ! -L "$MANIFEST_CHECKSUM_PATH" ]] || exit 1
if ! docker image inspect "$PROJECT_IMAGE" >/dev/null 2>&1; then
    exit 1
fi
[[ -d "$WORKSPACE_ROOT" && ! -L "$WORKSPACE_ROOT" ]] || exit 1
if [[ ! -d "$RUNS_ROOT" ]]; then
    mkdir "$RUNS_ROOT"
fi
[[ -d "$RUNS_ROOT" && ! -L "$RUNS_ROOT" ]] || exit 1
mkdir "$run_dir"
[[ -d "$run_dir" && ! -L "$run_dir" ]] || exit 1
chmod 700 "$run_dir"
if ! create_owner_marker; then
    exit 1
fi
if ! validate_owned_run; then
    exit 1
fi
run_dir_created=1
if ! (
    set -C
    printf '%s\n' "$preflight_json" > "$run_dir/preflight.json"
); then
    exit 1
fi

case "$model" in
    laya) model_arg=laya ;;
    qwen) model_arg=qwen ;;
    router) model_arg=laya-router ;;
esac
if ! validate_owned_run; then
    exit 1
fi

docker_command=(
    docker run --rm --pull=never --cidfile "$CONTAINER_CIDFILE"
    --name "$container_name"
    --label com.sysone-bench.owned=true
    --label "com.sysone-bench.run-id=$run_id"
    --cpus=4
    --memory=12g
    --memory-swap=12g
    --cpuset-cpus=0-3
    --user "$HOST_UID:$HOST_GID"
    -v "$run_dir:/results/$run_id"
    -v "$MODEL_CACHE:/models"
    -v "$MANIFEST_PATH:/input/manifest.jsonl:ro"
    -v "$MANIFEST_CHECKSUM_PATH:/input/manifest.sha256:ro"
    -w /workspace
    -e PYTHONUNBUFFERED=1
    -e HF_HOME=/models/huggingface
    -e XDG_CACHE_HOME=/models/cache
    -e TRANSFORMERS_CACHE=/models/huggingface/hub
    "$PROJECT_IMAGE"
    /usr/bin/nice -n 19
    /usr/bin/ionice -c 3
    "$VENV_PYTHON"
    -m "$V2_MODULE"
    --models "$model_arg"
    --output-root "/results/$run_id"
    --manifest /input/manifest.jsonl
    --manifest-checksum /input/manifest.sha256
    --run-id "$run_id"
)
"${docker_command[@]}"
if ! container_id="$(read_container_id)"; then
    exit 1
fi

if ! record_postflight; then
    exit 1
fi

required_artifacts=(metadata.json predictions.jsonl summary.json usage.json)
for artifact in "${required_artifacts[@]}"; do
    if [[ ! -f "$run_dir/$artifact" ]]; then
        printf '%s\n' "model run did not produce a regular required artifact: $artifact" >&2
        exit 1
    fi
    if [[ -L "$run_dir/$artifact" ]]; then
        printf '%s\n' "model run produced a symlinked artifact: $artifact" >&2
        exit 1
    fi
done
if [[ -n "$(find "$run_dir" -mindepth 1 -type l -print -quit)" ]]; then
    printf '%s\n' "model run produced a symlink in the result directory" >&2
    exit 1
fi
(
    cd "$WORKSPACE_ROOT"
    "$PREFLIGHT_PYTHON" -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from benchmark.storage import write_checksums; write_checksums(Path(sys.argv[2]))' \
        "$WORKSPACE_ROOT" "$run_dir"
)
printf '%s\n' "$run_dir"
