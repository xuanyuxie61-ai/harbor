#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/208_synth_project_Advanced"

if [[ ! -f "$ADVANCED_DIR/main.py" ]]; then
  echo "error: missing $ADVANCED_DIR/main.py" >&2
  exit 1
fi

cat > "$DIR/executable" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/208_synth_project_Advanced"

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Synthesis Python project 208

Usage:
  ./executable [arguments passed to main.py]
  ./executable --help
  ./executable --version

The executable is a local wrapper around 208_synth_project_Advanced/main.py. With no
arguments it runs the project's original demonstration pipeline.
HELP
    exit 0
    ;;
  --version)
    echo "synthesis-python-208 1.0"
    exit 0
    ;;
esac

cd "$ADVANCED_DIR"
export PYTHONPATH="$ADVANCED_DIR${PYTHONPATH:+:$PYTHONPATH}"


exec python3 "$ADVANCED_DIR/main.py" "$@"
EOF
chmod +x "$DIR/executable"
