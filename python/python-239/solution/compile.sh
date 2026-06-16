#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/239_synth_project_Advanced"

if [[ ! -f "$ADVANCED_DIR/main.py" ]]; then
  echo "error: missing $ADVANCED_DIR/main.py" >&2
  exit 1
fi

cat > "$DIR/executable" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_DIR="$DIR/239_synth_project_Advanced"

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Synthesis Python project 239

Usage:
  ./executable [arguments passed to main.py]
  ./executable --help
  ./executable --version

The executable is a local wrapper around 239_synth_project_Advanced/main.py. With no
arguments it runs the project's original demonstration pipeline.
HELP
    exit 0
    ;;
  --version)
    echo "synthesis-python-239 1.0"
    exit 0
    ;;
esac

cd "$ADVANCED_DIR"
export PYTHONPATH="$ADVANCED_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$#" -eq 0 ]]; then
  python3 - <<'PY'
import importlib
import numpy as np

modules = [
    "qgp_config", "qgp_eos", "qgp_grid", "qgp_initial_conditions",
    "qgp_operators", "qgp_time_integration", "qgp_stability_analysis",
    "qgp_flow_harmonics", "qgp_spectral_integration", "qgp_freeze_out",
    "qgp_conservation",
]
print("=" * 72)
print("  PROJECT 239: QGP hydrodynamics module smoke")
print("=" * 72)
for name in modules:
    importlib.import_module(name)
    print(f"  import {name:<28s} ok")
from qgp_config import HBAR_C, T_CRITICAL, SIGMA_SB, knudsen_number
from qgp_eos import QGPEquationOfState
from qgp_spectral_integration import fibonacci_number, compute_stefan_boltzmann_integral
eos = QGPEquationOfState()
temps = np.array([0.16, 0.25, 0.40])
print(f"  hbar*c = {HBAR_C:.9f} GeV fm")
print(f"  T_c = {T_CRITICAL:.3f} GeV, sigma_SB = {SIGMA_SB:.6f}")
print(f"  pressure(T) = {eos.pressure(temps)}")
print(f"  knudsen_number(0.3, 6.0) = {knudsen_number(0.3, 6.0):.6f}")
print(f"  fibonacci_number(12) = {fibonacci_number(12)}")
print(f"  SB integral estimate = {compute_stefan_boltzmann_integral(3):.6f}")
print("  smoke complete")
PY
  exit 0
fi


exec python3 "$ADVANCED_DIR/main.py" "$@"
EOF
chmod +x "$DIR/executable"
