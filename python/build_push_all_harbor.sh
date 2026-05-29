#!/usr/bin/env bash
# Build and push Harbor Python task images to Aliyun ACR.
#
# Run this on the ECS machine after syncing harbor/python to /root/sci-swe/harbor/python.
# It scans existing python-XXX directories, builds each Docker image from the
# task root, tags it for ACR, pushes it, and records per-task logs.

set -Eeuo pipefail

ACR="${ACR:-agent-rl-acr-registry.cn-beijing.cr.aliyuncs.com}"
ACR_VPC="${ACR_VPC:-agent-rl-acr-registry-vpc.cn-beijing.cr.aliyuncs.com}"
NAMESPACE="${NAMESPACE:-sci-swe-python}"
TASKS_DIR="${TASKS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOG_DIR="${LOG_DIR:-$TASKS_DIR/build-logs}"
START=1
END=999
DRY_RUN=0
NO_CACHE=0
KEEP_IMAGES=0
PUSH_RETRIES=3
JOBS=1
PLATFORM=""
ONLY_LIST=""
SKIP_LIST=""

usage() {
  cat <<'EOF'
Usage:
  build_push_all_harbor.sh [options]

Options:
  --tasks-dir PATH      Harbor python dataset dir. Default: script directory.
  --acr HOST            Public ACR host used for push.
  --namespace NAME      ACR namespace. Default: sci-swe-python.
  --start N             First task number to include. Default: 1.
  --end N               Last task number to include. Default: 999.
  --only LIST           Comma-separated task numbers/names, e.g. 004,045,python-200.
  --skip LIST           Comma-separated task numbers/names to skip, e.g. 001,002.
  --platform VALUE      Optional docker build platform, e.g. linux/amd64.
  --push-retries N      Docker push retry count. Default: 3.
  --jobs N              Number of tasks to build/push in parallel. Default: 1.
  --no-cache            Pass --no-cache to docker build.
  --keep-images         Do not remove local images after each push.
  --dry-run             Print planned docker commands without build/push.
  -h, --help            Show this help.

Examples:
  # Upload all existing task directories.
  bash build_push_all_harbor.sh

  # python-001 and python-002 are already uploaded.
  bash build_push_all_harbor.sh --skip 001,002

  # Use 4-way parallel build/push.
  bash build_push_all_harbor.sh --jobs 4

  # Upload only selected tasks.
  bash build_push_all_harbor.sh --only 004,045,200

Environment overrides:
  ACR, ACR_VPC, NAMESPACE, TASKS_DIR, LOG_DIR
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

task_number() {
  local task="$1"
  local num="${task#python-}"
  printf '%d' "$((10#$num))"
}

log() {
  local line="$*"
  if command -v flock >/dev/null 2>&1; then
    {
      flock -x 200
      echo "$line" | tee -a "$SUMMARY_LOG"
    } 200>"$SUMMARY_LOG.lock"
  else
    echo "$line" | tee -a "$SUMMARY_LOG"
  fi
}

dry_run_line() {
  printf '[DRY-RUN]'
  printf ' %q' "$@"
  printf '\n'
}

write_status() {
  local task_name="$1"
  local state="$2"
  local message="$3"
  printf '%s|%s|%s\n' "$state" "$task_name" "$message" > "$STATUS_DIR/$task_name.status"
}

validate_task() {
  local task_dir="$1"
  local task_name="$2"
  local missing=0
  local rel
  local required=(
    "instruction.md"
    "task.toml"
    "environment/Dockerfile"
    "environment/workspace/main.py"
    "tests/test.sh"
    "tests/test_main.py"
  )

  for rel in "${required[@]}"; do
    if [[ ! -s "$task_dir/$rel" ]]; then
      echo "missing or empty: $rel"
      missing=1
    fi
  done

  if [[ ! -d "$task_dir/environment/workspace" ]]; then
    echo "missing dir: environment/workspace"
    missing=1
  elif ! find "$task_dir/environment/workspace" -type f -name '*.py' | grep -q .; then
    echo "workspace has no Python source files"
    missing=1
  fi

  if [[ -s "$task_dir/task.toml" ]] \
    && ! grep -q "name = \"sci-swe/$task_name\"" "$task_dir/task.toml"; then
    echo "task.toml task name mismatch"
    missing=1
  fi

  if [[ -s "$task_dir/environment/Dockerfile" ]]; then
    if ! grep -q 'COPY environment/workspace' "$task_dir/environment/Dockerfile"; then
      echo "Dockerfile must copy environment/workspace from the task root"
      missing=1
    fi
    if ! grep -q 'COPY tests' "$task_dir/environment/Dockerfile"; then
      echo "Dockerfile must copy tests into the image"
      missing=1
    fi
  fi

  return "$missing"
}

docker_push_with_retry() {
  local image="$1"
  local task_log="$2"
  local attempt

  for attempt in $(seq 1 "$PUSH_RETRIES"); do
    if docker push "$image" >> "$task_log" 2>&1; then
      return 0
    fi
    echo "push attempt $attempt/$PUSH_RETRIES failed: $image" >> "$task_log"
    sleep $((attempt * 5))
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks-dir)
      TASKS_DIR="$2"
      shift 2
      ;;
    --acr)
      ACR="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
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
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --push-retries)
      PUSH_RETRIES="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --keep-images)
      KEEP_IMAGES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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
[[ "$PUSH_RETRIES" =~ ^[0-9]+$ ]] || die "--push-retries must be numeric"
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs must be numeric"
(( JOBS >= 1 )) || die "--jobs must be at least 1"

mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"
SUCCESS_LIST="$LOG_DIR/success.txt"
FAILED_LIST="$LOG_DIR/failed.txt"
SKIPPED_LIST="$LOG_DIR/skipped.txt"
STATUS_DIR="$LOG_DIR/status"
: > "$SUMMARY_LOG"
: > "$SUCCESS_LIST"
: > "$FAILED_LIST"
: > "$SKIPPED_LIST"
mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v docker >/dev/null 2>&1 || die "docker is not installed or not in PATH"
fi

mapfile -t TASK_DIRS < <(find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d -name 'python-[0-9][0-9][0-9]' | sort)
[[ "${#TASK_DIRS[@]}" -gt 0 ]] || die "no python-XXX directories found under $TASKS_DIR"

SELECTED=()
for task_dir in "${TASK_DIRS[@]}"; do
  task_name="$(basename "$task_dir")"
  number="$(task_number "$task_name")"

  if [[ -n "$ONLY_LIST" ]]; then
    contains_csv_task "$ONLY_LIST" "$task_name" || continue
  else
    (( number >= START && number <= END )) || continue
  fi

  if contains_csv_task "$SKIP_LIST" "$task_name"; then
    printf '%s skipped by --skip\n' "$task_name" >> "$SKIPPED_LIST"
    continue
  fi

  SELECTED+=("$task_dir")
done

TOTAL="${#SELECTED[@]}"
[[ "$TOTAL" -gt 0 ]] || die "no tasks selected"

log "========================================"
log "Harbor Python build and push"
log "tasks_dir=$TASKS_DIR"
log "acr=$ACR"
log "namespace=$NAMESPACE"
log "vpc_pull_prefix=$ACR_VPC/$NAMESPACE"
log "selected_tasks=$TOTAL"
log "dry_run=$DRY_RUN"
log "jobs=$JOBS"
log "started_at=$(date -Is)"
log "========================================"

process_task() {
  local index="$1"
  local task_dir="$2"
  local task_name dockerfile local_tag acr_tag task_log validation_output
  local -a build_cmd
  local line

  task_name="$(basename "$task_dir")"
  dockerfile="$task_dir/environment/Dockerfile"
  local_tag="harbor-${task_name}:latest"
  acr_tag="$ACR/$NAMESPACE/$task_name:latest"
  task_log="$LOG_DIR/$task_name.log"
  : > "$task_log"

  log ""
  log "[$index/$TOTAL] $task_name"
  log "  push: $acr_tag"
  log "  pull in ACS: $ACR_VPC/$NAMESPACE/$task_name:latest"
  echo "started_at=$(date -Is)" >> "$task_log"
  echo "task=$task_name" >> "$task_log"
  echo "push=$acr_tag" >> "$task_log"

  if ! validation_output="$(validate_task "$task_dir" "$task_name" 2>&1)"; then
    log "  [SKIP] invalid task layout"
    printf '%s\n%s\n\n' "$task_name" "$validation_output" >> "$task_log"
    write_status "$task_name" "skipped" "invalid task layout, log=$task_log"
    return 0
  fi

  build_cmd=(docker build -t "$local_tag" -f "$dockerfile")
  [[ "$NO_CACHE" -eq 1 ]] && build_cmd+=(--no-cache)
  [[ -n "$PLATFORM" ]] && build_cmd+=(--platform "$PLATFORM")
  build_cmd+=("$task_dir")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    line="$(dry_run_line "${build_cmd[@]}")"
    echo "$line" >> "$task_log"
    log "$line"
    line="$(dry_run_line docker tag "$local_tag" "$acr_tag")"
    echo "$line" >> "$task_log"
    log "$line"
    line="$(dry_run_line docker push "$acr_tag")"
    echo "$line" >> "$task_log"
    log "$line"
    write_status "$task_name" "success" "$acr_tag"
    return 0
  fi

  if "${build_cmd[@]}" >> "$task_log" 2>&1; then
    log "  [BUILD OK]"
  else
    log "  [BUILD FAIL] see $task_log"
    write_status "$task_name" "failed" "build failed, log=$task_log"
    return 0
  fi

  if docker tag "$local_tag" "$acr_tag" >> "$task_log" 2>&1; then
    log "  [TAG OK]"
  else
    log "  [TAG FAIL] see $task_log"
    write_status "$task_name" "failed" "tag failed, log=$task_log"
    return 0
  fi

  if docker_push_with_retry "$acr_tag" "$task_log"; then
    log "  [PUSH OK]"
    write_status "$task_name" "success" "$acr_tag"
  else
    log "  [PUSH FAIL] see $task_log"
    write_status "$task_name" "failed" "push failed, log=$task_log"
  fi

  if [[ "$KEEP_IMAGES" -eq 0 ]]; then
    docker rmi "$local_tag" "$acr_tag" >> "$task_log" 2>&1 || true
    if [[ "$JOBS" -eq 1 ]]; then
      docker image prune -f >> "$task_log" 2>&1 || true
    fi
  fi
  echo "finished_at=$(date -Is)" >> "$task_log"
  return 0
}

INDEX=0
RUNNING=0

for task_dir in "${SELECTED[@]}"; do
  INDEX=$((INDEX + 1))
  process_task "$INDEX" "$task_dir" &
  RUNNING=$((RUNNING + 1))

  if (( RUNNING >= JOBS )); then
    wait -n
    RUNNING=$((RUNNING - 1))
  fi
done

wait

OK=0
FAIL=0
SKIP=0
shopt -s nullglob
for status_file in "$STATUS_DIR"/*.status; do
  IFS='|' read -r state task_name message < "$status_file"
  case "$state" in
    success)
      OK=$((OK + 1))
      printf '%s %s\n' "$task_name" "$message" >> "$SUCCESS_LIST"
      ;;
    failed)
      FAIL=$((FAIL + 1))
      printf '%s %s\n' "$task_name" "$message" >> "$FAILED_LIST"
      ;;
    skipped)
      SKIP=$((SKIP + 1))
      printf '%s %s\n' "$task_name" "$message" >> "$SKIPPED_LIST"
      ;;
  esac
done
shopt -u nullglob

log ""
log "========================================"
log "finished_at=$(date -Is)"
log "success=$OK failed=$FAIL skipped=$SKIP total=$TOTAL"
log "summary_log=$SUMMARY_LOG"
log "success_list=$SUCCESS_LIST"
log "failed_list=$FAILED_LIST"
log "skipped_list=$SKIPPED_LIST"
log "========================================"

if [[ "$FAIL" -gt 0 || "$SKIP" -gt 0 ]]; then
  exit 1
fi
