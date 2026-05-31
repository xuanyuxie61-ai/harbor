#!/usr/bin/env bash
# Delete stale Aliyun ACR tags for Harbor Python task images.
#
# The script compares local python-NNN directories with an expected numeric
# range. Tags whose local task directory no longer exists are deleted from ACR.
# It is dry-run by default; pass --apply to really delete remote tags.

set -Eeuo pipefail

TASKS_DIR="${TASKS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ACR_REGION="${ACR_REGION:-cn-beijing}"
INSTANCE_ID="${INSTANCE_ID:-cri-h0hcfm1j42x9wiw6}"
NAMESPACE="${NAMESPACE:-sci-swe-python}"
TAG="${TAG:-latest}"
START=1
END=200
DRY_RUN=1
ONLY_LIST=""
SKIP_LIST="003,076,090"

usage() {
  cat <<'EOF'
Usage:
  delete_stale_acr_tags.sh [options]

Options:
  --tasks-dir PATH      Local Harbor python dataset dir. Default: script directory.
  --region NAME         Aliyun region. Default: cn-beijing.
  --instance-id ID      ACR Enterprise instance ID.
  --namespace NAME      ACR namespace. Default: sci-swe-python.
  --tag TAG             Tag to delete. Default: latest.
  --start N             First task number to check. Default: 1.
  --end N               Last task number to check. Default: 200.
  --only LIST           Comma-separated task numbers/names to delete/check.
  --skip LIST           Comma-separated task numbers/names to skip.
  --apply               Really delete remote ACR tags. Default is dry-run.
  -h, --help            Show this help.

Examples:
  # Preview stale tags whose local python-NNN directory no longer exists.
  bash delete_stale_acr_tags.sh

  # Really delete stale latest tags from ACR.
  bash delete_stale_acr_tags.sh --apply

  # Delete only selected tags.
  bash delete_stale_acr_tags.sh --only 002,011,018 --apply

Environment overrides:
  TASKS_DIR, ACR_REGION, INSTANCE_ID, NAMESPACE, TAG
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_task_name() {
  local item="$1"
  item="${item#python-}"
  if [[ ! "$item" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf 'python-%03d' "$((10#$item))"
}

contains_csv_task() {
  local csv="$1"
  local task="$2"
  local item normalized

  [[ -n "$csv" ]] || return 1
  IFS=',' read -ra parts <<< "$csv"
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || continue
    normalized="$(normalize_task_name "$item")" || continue
    [[ "$normalized" == "$task" ]] && return 0
  done
  return 1
}

dry_run_line() {
  printf '[DRY-RUN]'
  printf ' %q' "$@"
  printf '\n'
}

extract_json_field() {
  local field="$1"
  python3 -c '
import json
import sys

field = sys.argv[1]
data = json.load(sys.stdin)
value = data.get(field)
if value is None:
    data = data.get("Data", {})
    value = data.get(field)
if value is None:
    data = data.get("Repository", {})
    value = data.get(field)
if value is None:
    sys.exit(1)
print(value)
' "$field"
}

get_repo_id() {
  local task_name="$1"
  local output repo_id

  output="$(
    aliyun cr GetRepository \
      --InstanceId "$INSTANCE_ID" \
      --RepoNamespaceName "$NAMESPACE" \
      --RepoName "$task_name" \
      --region "$ACR_REGION" \
      2>/dev/null
  )" || return 1

  repo_id="$(printf '%s' "$output" | extract_json_field RepoId)" || return 1
  [[ -n "$repo_id" ]] || return 1
  printf '%s\n' "$repo_id"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks-dir)
      TASKS_DIR="$2"
      shift 2
      ;;
    --region)
      ACR_REGION="$2"
      shift 2
      ;;
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --start)
      START="$2"
      shift 2
      ;;
    --end)
      END="$2"
      shift 2
      ;;
    --only)
      ONLY_LIST="$2"
      shift 2
      ;;
    --skip)
      SKIP_LIST="$2"
      shift 2
      ;;
    --apply)
      DRY_RUN=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -d "$TASKS_DIR" ]] || die "TASKS_DIR does not exist: $TASKS_DIR"
[[ "$START" =~ ^[0-9]+$ ]] || die "--start must be numeric"
[[ "$END" =~ ^[0-9]+$ ]] || die "--end must be numeric"
(( START <= END )) || die "--start must be <= --end"

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v aliyun >/dev/null 2>&1 || die "aliyun CLI is not installed or not in PATH"
fi

DELETE_TASKS=()
for number in $(seq "$START" "$END"); do
  task_name="$(printf 'python-%03d' "$number")"

  if [[ -n "$ONLY_LIST" ]]; then
    contains_csv_task "$ONLY_LIST" "$task_name" || continue
  elif [[ -d "$TASKS_DIR/$task_name" ]]; then
    continue
  fi

  if contains_csv_task "$SKIP_LIST" "$task_name"; then
    continue
  fi

  DELETE_TASKS+=("$task_name")
done

echo "========================================"
echo "ACR stale tag cleanup"
echo "tasks_dir=$TASKS_DIR"
echo "region=$ACR_REGION"
echo "instance_id=$INSTANCE_ID"
echo "namespace=$NAMESPACE"
echo "tag=$TAG"
echo "range=$START..$END"
echo "skip=$SKIP_LIST"
echo "dry_run=$DRY_RUN"
echo "delete_candidates=${#DELETE_TASKS[@]}"
echo "========================================"

if [[ "${#DELETE_TASKS[@]}" -eq 0 ]]; then
  echo "No stale tags selected."
  exit 0
fi

OK=0
FAIL=0
for task_name in "${DELETE_TASKS[@]}"; do
  if [[ "$DRY_RUN" -eq 1 ]]; then
    dry_run_line aliyun cr GetRepository --InstanceId "$INSTANCE_ID" --RepoNamespaceName "$NAMESPACE" --RepoName "$task_name" --region "$ACR_REGION"
    dry_run_line aliyun cr DeleteRepoTag --InstanceId "$INSTANCE_ID" --RepoId "<RepoId for $task_name>" --Tag "$TAG" --region "$ACR_REGION"
    OK=$((OK + 1))
    continue
  fi

  echo "[DELETE] $NAMESPACE/$task_name:$TAG"
  if ! repo_id="$(get_repo_id "$task_name")"; then
    echo "[FAIL] $NAMESPACE/$task_name:$TAG could not resolve RepoId" >&2
    FAIL=$((FAIL + 1))
    continue
  fi

  cmd=(
    aliyun cr DeleteRepoTag
    --InstanceId "$INSTANCE_ID"
    --RepoId "$repo_id"
    --Tag "$TAG"
    --region "$ACR_REGION"
  )

  if "${cmd[@]}"; then
    OK=$((OK + 1))
  else
    echo "[FAIL] $NAMESPACE/$task_name:$TAG" >&2
    FAIL=$((FAIL + 1))
  fi
done

echo "========================================"
echo "finished ok=$OK failed=$FAIL total=${#DELETE_TASKS[@]}"
echo "========================================"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
