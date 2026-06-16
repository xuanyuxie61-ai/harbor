#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/292_synth_project_Advanced"

if [[ ! -f "$ADVANCED_DIR/main.py" ]]; then
  echo "error: missing $ADVANCED_DIR/main.py" >&2
  exit 1
fi

cat > "$DIR/executable" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/292_synth_project_Advanced"

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Synthesis Python project 292

Usage:
  ./executable [arguments passed to main.py]
  ./executable --help
  ./executable --version

The executable is a local wrapper around 292_synth_project_Advanced/main.py. With no
arguments it runs the project's original demonstration pipeline.
HELP
    exit 0
    ;;
  --version)
    echo "synthesis-python-292 1.0"
    exit 0
    ;;
esac

cd "$ADVANCED_DIR"
export PYTHONPATH="$ADVANCED_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$#" -eq 0 ]]; then
  python3 - <<'PY'
import importlib

modules = [
    "mesh_generator", "high_order_fd", "resistive_mhd",
    "stability_analysis", "current_sheet", "spectral_laguerre",
]
print("=" * 72)
print("  PROJECT 292: magnetic reconnection module smoke")
print("=" * 72)
for name in modules:
    module = importlib.import_module(name)
    exported = [k for k in dir(module) if not k.startswith("_")][:6]
    print(f"  import {name:<20s} ok  exports={exported}")
from mesh_generator import MeshGenerator
from high_order_fd import HighOrderFD
mesh = MeshGenerator(nx=16, ny=8, x_range=(-1.0, 1.0), y_range=(-0.5, 0.5))
mesh.build_structured_mesh()
mesh.compute_metrics()
fd = HighOrderFD(order=4, nx=mesh.nx, dx=mesh.dx_uniform)
fd.build_central_stencils()
print(f"  mesh: {mesh.nx} x {mesh.ny}, dx_min={mesh.dx_min:.6f}, dy_min={mesh.dy_min:.6f}")
print(f"  fd: order={fd.order}, half_width={fd.half_width}")
print("  smoke complete")
PY
  exit 0
fi


exec python3 "$ADVANCED_DIR/main.py" "$@"
EOF
chmod +x "$DIR/executable"
