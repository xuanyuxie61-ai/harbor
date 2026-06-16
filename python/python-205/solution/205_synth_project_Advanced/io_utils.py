"""
io_utils.py - Structured Data I/O and Batch Simulation Management

This module handles data persistence, structured file parsing,
and batch simulation orchestration for the UQ pipeline. It integrates:
  - Scenario-based simulation batching (from sbrc-2014-simulation)
  - Structured data file parsing (from gpl_display)
  - Parameter serialization and result archival

Mathematical Context
--------------------
For surrogate-based UQ, the I/O layer must handle:
1. Training data: (xi_i, y_i) pairs where xi_i in R^d, y_i in R^q
2. Surrogate metadata: polynomial degrees, basis types, coefficient tensors
3. Validation results: cross-validation scores, error statistics
4. Sensitivity indices: Sobol first-order and total-order indices

Data formats follow a structured text-based approach for reproducibility.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional, Any
import os
import json
import math


# ---------------------------------------------------------------------------
# Structured data file I/O
# ---------------------------------------------------------------------------

class StructuredDataWriter:
    """
    Writer for structured numerical data files.

    File format:
      # Comment lines start with #
      # METADATA key=value
      DIMENSION ndim noutput
      NSAMPLES n
      DATA
      xi_1 ... xi_d y_1 ... y_q
      ...

    This format is parseable by the StructuredDataReader and
    supports both curve data (single points) and grid data
    (tensor product layouts).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, str] = {}
        self.data_blocks: List[Dict[str, Any]] = []

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = str(value)

    def add_data_block(self, name: str, x: NDArray,
                       y: Optional[NDArray] = None):
        """
        Add a data block.

        Parameters
        ----------
        name : str
            Block identifier.
        x : ndarray(n,) or ndarray(n, d)
            Input coordinates.
        y : ndarray(n,) or ndarray(n, q), optional
            Output values.
        """
        block = {'name': name, 'x': np.atleast_2d(x)}
        if y is not None:
            block['y'] = np.atleast_2d(y)
        self.data_blocks.append(block)

    def write(self):
        """Write all data to file."""
        with open(self.filepath, 'w') as f:
            f.write("# Structured UQ Data File\n")
            f.write(f"# FORMAT_VERSION 1.0\n")
            for key, val in self.metadata.items():
                f.write(f"# META {key}={val}\n")
            f.write(f"# NBLOCKS {len(self.data_blocks)}\n")

            for block in self.data_blocks:
                f.write(f"\n# BLOCK {block['name']}\n")
                x = block['x']
                if x.ndim == 1:
                    x = x.reshape(-1, 1)
                n, d = x.shape

                if 'y' in block:
                    y = block['y']
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    q = y.shape[1]
                    f.write(f"# DIMENSION {d} {q}\n")
                    f.write(f"# NSAMPLES {n}\n")
                    for i in range(n):
                        row = list(x[i]) + list(y[i])
                        f.write(" ".join(f"{v:.12e}" for v in row) + "\n")
                else:
                    f.write(f"# DIMENSION {d} 0\n")
                    f.write(f"# NSAMPLES {n}\n")
                    for i in range(n):
                        row = list(x[i])
                        f.write(" ".join(f"{v:.12e}" for v in row) + "\n")


class StructuredDataReader:
    """
    Reader for structured numerical data files.

    Parses the format written by StructuredDataWriter.
    Handles blank-line separators and comment lines.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, str] = {}
        self.data_blocks: Dict[str, Dict[str, NDArray]] = {}
        self._parse()

    def _parse(self):
        """Parse the data file."""
        if not os.path.exists(self.filepath):
            return

        current_block = None
        current_data: List[List[float]] = []
        dim_in = 0
        dim_out = 0

        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Blank line: finalize current block
                    if current_block and current_data:
                        arr = np.array(current_data)
                        self.data_blocks[current_block] = {
                            'x': arr[:, :dim_in],
                            'y': arr[:, dim_in:dim_in + dim_out] if dim_out > 0 else None
                        }
                    current_block = None
                    current_data = []
                    continue

                if line.startswith('# META '):
                    kv = line[7:].split('=', 1)
                    if len(kv) == 2:
                        self.metadata[kv[0]] = kv[1]
                elif line.startswith('# BLOCK '):
                    current_block = line[8:].strip()
                    current_data = []
                elif line.startswith('# DIMENSION '):
                    parts = line[12:].split()
                    dim_in = int(parts[0])
                    dim_out = int(parts[1]) if len(parts) > 1 else 0
                elif line.startswith('#'):
                    continue
                else:
                    try:
                        values = [float(v) for v in line.split()]
                        current_data.append(values)
                    except ValueError:
                        pass

        # Finalize last block
        if current_block and current_data:
            arr = np.array(current_data)
            self.data_blocks[current_block] = {
                'x': arr[:, :dim_in],
                'y': arr[:, dim_in:dim_in + dim_out] if dim_out > 0 else None
            }

    def get_block(self, name: str) -> Optional[Dict[str, NDArray]]:
        """Retrieve a named data block."""
        return self.data_blocks.get(name)

    def list_blocks(self) -> List[str]:
        """List all block names."""
        return list(self.data_blocks.keys())


# ---------------------------------------------------------------------------
# Batch simulation management
# ---------------------------------------------------------------------------

class SimulationBatchManager:
    """
    Manages batch execution of expensive truth model simulations.

    Organizes simulation scenarios as combinations of:
      - Parameter sets (from DOE)
      - Physics model configurations
      - Output specifications

    This follows the scenario-based approach from sbrc-2014-simulation
    where multiple trace/algorithm/host combinations are enumerated
    and executed systematically.

    Attributes
    ----------
    scenarios : list of dict
        Each scenario specifies parameter values and configuration.
    results : dict
        Mapping from scenario_id to simulation results.
    """

    def __init__(self, base_dir: str = '.'):
        self.base_dir = base_dir
        self.scenarios: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self._scenario_counter = 0

    def add_scenario(self, params: Dict[str, float],
                     model_config: Optional[Dict] = None,
                     label: str = '') -> str:
        """
        Register a new simulation scenario.

        Parameters
        ----------
        params : dict
            Parameter values for this scenario.
        model_config : dict, optional
            Model-specific configuration.
        label : str
            Human-readable label.

        Returns
        -------
        str
            Unique scenario ID.
        """
        sid = f"scenario_{self._scenario_counter:06d}"
        self._scenario_counter += 1
        self.scenarios.append({
            'id': sid,
            'params': params.copy(),
            'config': model_config or {},
            'label': label
        })
        return sid

    def enumerate_scenarios(self, param_grid: Dict[str, NDArray],
                            model_configs: Optional[List[Dict]] = None):
        """
        Enumerate all scenarios from parameter grid (tensor product).

        Parameters
        ----------
        param_grid : dict
            Maps parameter names to arrays of values.
        model_configs : list of dict, optional
            List of model configurations to cross with parameter grid.
        """
        param_names = list(param_grid.keys())
        param_values = [np.atleast_1d(param_grid[k]) for k in param_names]

        # Tensor product enumeration
        configs = model_configs or [{}]

        def recurse(idx: int, current_params: Dict):
            if idx == len(param_names):
                for ci, cfg in enumerate(configs):
                    label = f"config_{ci}_" + "_".join(
                        f"{k}={current_params[k]:.4g}" for k in param_names
                    )
                    self.add_scenario(current_params.copy(), cfg, label)
                return
            for v in param_values[idx]:
                current_params[param_names[idx]] = float(v)
                recurse(idx + 1, current_params)

        recurse(0, {})

    def store_result(self, scenario_id: str, result: Dict[str, Any]):
        """Store result for a scenario."""
        self.results[scenario_id] = result

    def get_result(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve result for a scenario."""
        return self.results.get(scenario_id)

    def get_training_data(self) -> Tuple[NDArray, NDArray]:
        """
        Extract training data arrays from stored results.

        Returns
        -------
        X : ndarray(n_samples, d)
            Input parameters.
        Y : ndarray(n_samples, q)
            Output QoIs.
        """
        X_list = []
        Y_list = []
        for scenario in self.scenarios:
            sid = scenario['id']
            if sid in self.results:
                result = self.results[sid]
                X_list.append(list(scenario['params'].values()))
                if 'qoi' in result:
                    Y_list.append(np.atleast_1d(result['qoi']))
                elif 'outputs' in result:
                    Y_list.append(np.atleast_1d(result['outputs']))

        if not X_list:
            return np.array([[]]), np.array([[]])

        X = np.array(X_list)
        Y = np.array(Y_list)
        return X, Y

    def save_summary(self, filepath: str):
        """Save summary of all scenarios and results to file."""
        writer = StructuredDataWriter(filepath)
        writer.add_metadata('n_scenarios', len(self.scenarios))
        writer.add_metadata('n_completed', len(self.results))

        X, Y = self.get_training_data()
        if X.size > 0:
            writer.add_data_block('training_inputs', X)
            if Y.size > 0:
                writer.add_data_block('training_outputs', Y)

        writer.write()


# ---------------------------------------------------------------------------
# JSON-based metadata persistence
# ---------------------------------------------------------------------------

def save_surrogate_metadata(filepath: str, metadata: Dict[str, Any]):
    """
    Save surrogate model metadata to JSON file.

    Includes:
      - Basis type and degree
      - Multi-index set
      - Fitting method and regularization
      - Cross-validation scores
      - Input/output dimensions
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, val in metadata.items():
        if isinstance(val, np.ndarray):
            serializable[key] = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            serializable[key] = val.item()
        else:
            serializable[key] = val

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_surrogate_metadata(filepath: str) -> Dict[str, Any]:
    """Load surrogate metadata from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays where appropriate
    for key in ['multi_index', 'coefficients', 'nodes', 'weights']:
        if key in data and isinstance(data[key], list):
            data[key] = np.array(data[key])

    return data
