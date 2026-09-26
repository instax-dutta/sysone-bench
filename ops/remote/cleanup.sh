#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf '%s\n' "usage: cleanup.sh --run-id <id> --container-id <id>" >&2
    exit 2
}

run_id=""
container_id=""
while (($# > 0)); do
    case "$1" in
        --run-id)
            (($# >= 2)) || usage
            run_id="$2"
            shift 2
            ;;
        --container-id)
            (($# >= 2)) || usage
            container_id="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || usage
[[ "$run_id" != "." && "$run_id" != ".." ]] || usage
[[ "$container_id" =~ ^[0-9a-f]{12,64}$ ]] || exit 0
command -v docker >/dev/null 2>&1 || exit 0
if ! observed_id="$(docker container inspect --format '{{.Id}}' "$container_id" 2>/dev/null)"; then
    exit 0
fi
[[ "$observed_id" == "$container_id" ]] || exit 0
if ! owner_label="$(docker container inspect --format '{{ index .Config.Labels "com.sysone-bench.owned" }}' "$container_id" 2>/dev/null)"; then
    exit 0
fi
if ! run_label="$(docker container inspect --format '{{ index .Config.Labels "com.sysone-bench.run-id" }}' "$container_id" 2>/dev/null)"; then
    exit 0
fi
if [[ "$owner_label" == "true" && "$run_label" == "$run_id" ]]; then
    docker rm -f -- "$container_id" >/dev/null 2>&1 || true
fi
