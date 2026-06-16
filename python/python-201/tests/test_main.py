import json
import math
import os
import subprocess
from pathlib import Path


EXE = Path(os.environ.get("PROGRAM_UNDER_TEST", "/app/workspace/executable"))


def run_cmd(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(EXE), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )


def stdout(*args: str) -> str:
    return run_cmd(*args).stdout


def parse_json_output(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end >= start, text
    return json.loads(text[start : end + 1])


def json_cmd(*args: str) -> dict:
    return parse_json_output(stdout(*args, "--json"))


def assert_close(actual: float, expected: float, tol: float = 1e-12) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol)


def test_01_version_string():
    assert stdout("--version").strip() == "executable 201.1.0"


def test_02_help_lists_all_subcommands():
    text = stdout("--help")
    assert "usage: executable" in text
    for name in ["run", "config", "basis", "quadrature", "owen", "smoke"]:
        assert name in text


def test_03_config_text_summary():
    text = stdout("config")
    assert "dimensions=3" in text
    assert "basis_max_degree=4" in text
    assert "total_basis_dim=35" in text
    assert "grid=32x32" in text


def test_04_config_json_basis_fields():
    data = json_cmd("config")
    assert data["basis"]["max_degree"] == 4
    assert data["basis"]["truncation"] == "total_order"
    assert_close(data["basis"]["hyperbolic_q"], 0.5)
    assert data["total_basis_dim"] == 35


def test_05_config_json_cahn_hilliard_fields():
    data = json_cmd("config")["cahn_hilliard"]
    assert data["nx"] == 32
    assert data["ny"] == 32
    assert data["n_steps"] == 200
    assert_close(data["dt"], 1e-4)
    assert_close(data["kappa_mean"], 0.01)
    assert_close(data["gamma_mean"], 1.0)
    assert_close(data["mobility_mean"], 1.0)
    assert data["random_seed"] == 42


def test_06_config_json_measure_names_and_types():
    measures = json_cmd("config")["measures"]
    assert [m["name"] for m in measures] == ["kappa", "gamma", "mobility"]
    assert [m["type"] for m in measures] == ["gauss", "uniform", "gauss"]
    assert_close(measures[0]["mean"], 0.01)
    assert_close(measures[1]["variance"], 0.013333333333333327)
    assert_close(measures[2]["variance"], 0.0225)


def test_07_config_json_multi_element_and_sparse_grid():
    data = json_cmd("config")
    assert data["multi_element"]["initial_elements"] == 2
    assert data["multi_element"]["local_degree"] == 3
    assert data["multi_element"]["max_elements"] == 16
    assert data["sparse_grid"]["level"] == 4
    assert data["sparse_grid"]["growth_rule"] == "linear"
    assert data["sparse_grid"]["quadrature_type"] == "gauss-patterson"


def test_08_basis_degree_zero():
    data = json_cmd("basis", "--degree", "0")
    assert data["degree"] == 0
    assert data["n_dim"] == 3
    assert data["n_basis"] == 1
    assert data["indices"] == [[0, 0, 0]]


def test_09_basis_degree_one_indices():
    data = json_cmd("basis", "--degree", "1")
    assert data["n_basis"] == 4
    assert data["indices"] == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_10_basis_degree_two_indices_and_size():
    data = json_cmd("basis", "--degree", "2")
    assert data["n_basis"] == 10
    assert data["indices"][-1] == [2, 0, 0]
    assert all(sum(idx) <= 2 for idx in data["indices"])


def test_11_basis_degree_three_size_and_prefix():
    data = json_cmd("basis", "--degree", "3")
    assert data["degree"] == 3
    assert data["n_basis"] == 20
    assert data["truncation"] == "total_order"
    assert data["indices"][:4] == [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]


def test_12_basis_degree_four_limit_is_respected():
    data = json_cmd("basis", "--degree", "4", "--limit", "3")
    assert data["n_basis"] == 35
    assert len(data["indices"]) == 3
    assert data["indices"] == [[0, 0, 0], [0, 0, 1], [0, 0, 2]]


def test_13_basis_text_output():
    text = stdout("basis", "--degree", "2", "--limit", "4")
    assert "degree=2" in text
    assert "n_dim=3" in text
    assert "n_basis=10" in text
    assert "indices=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]]" in text


def test_14_quadrature_level_one_summary():
    data = json_cmd("quadrature", "--level", "1")
    assert data["level"] == 1
    assert data["n_points"] == 7
    assert_close(data["weight_sum"], 1.0)
    assert_close(data["weight_min"], -2.0)
    assert_close(data["weight_max"], 0.5)


def test_15_quadrature_level_two_summary():
    data = json_cmd("quadrature", "--level", "2")
    assert data["level"] == 2
    assert data["n_points"] == 25
    assert_close(data["weight_sum"], 1.0000000000000009)
    assert_close(data["weight_min"], -1.0)
    assert_close(data["weight_max"], 2.777777777777776)


def test_16_quadrature_level_two_first_nodes_with_limit():
    data = json_cmd("quadrature", "--level", "2", "--limit", "3")
    assert len(data["first_nodes"]) == 3
    assert data["first_nodes"][0] == [0.01, 1.0, 1.0]
    assert data["first_nodes"][1] == [0.01, 1.0, 0.85]
    assert data["first_nodes"][2] == [0.01, 1.0, 1.15]


def test_17_quadrature_level_three_summary():
    data = json_cmd("quadrature", "--level", "3", "--limit", "3")
    assert data["level"] == 3
    assert data["n_points"] == 69
    assert_close(data["weight_sum"], 1.0000000000000022, tol=1e-11)
    assert data["weight_min"] < 0
    assert data["weight_max"] > 1


def test_18_quadrature_text_output():
    text = stdout("quadrature", "--level", "2")
    assert "level=2" in text
    assert "n_points=25" in text
    assert "weight_sum=1.000000000000e+00" in text


def test_19_owen_reference_value():
    data = json_cmd("owen", "--h", "1", "--a", "0.5")
    assert data["h"] == 1.0
    assert data["a"] == 0.5
    assert_close(data["value"], 0.043064691060608055, tol=1e-14)


def test_20_owen_zero_h_unit_a():
    data = json_cmd("owen", "--h", "0", "--a", "1")
    assert_close(data["value"], 0.1249999994546938, tol=1e-12)


def test_21_owen_half_h_unit_a():
    data = json_cmd("owen", "--h", "0.5", "--a", "1")
    assert_close(data["value"], 0.10667106241614152, tol=1e-14)


def test_22_owen_large_reference():
    data = json_cmd("owen", "--h", "2", "--a", "2")
    assert_close(data["value"], 0.011377972024179922, tol=1e-14)


def test_23_owen_text_output():
    text = stdout("owen", "--h", "1", "--a", "0.5")
    assert text.strip() == "T(1, 0.5) = 4.306469106061e-02"


def test_24_smoke_default_shape_and_counts():
    data = json_cmd("smoke")
    assert data["basis_dim"] == 10
    assert data["grid"] == [8, 8]
    assert data["sparse_points"] == 25
    assert data["steps"] == 2


def test_25_smoke_default_conservation_and_energy():
    data = json_cmd("smoke")
    assert_close(data["initial_mass"], -0.003288182315596044, tol=1e-15)
    assert_close(data["final_mass"], data["initial_mass"], tol=1e-15)
    assert data["mass_error"] == 0.0
    assert_close(data["energy_initial"], 0.25061488565649115, tol=1e-14)
    assert_close(data["energy_final"], 0.2503417709597431, tol=1e-14)
    assert data["energy_final"] < data["energy_initial"]


def test_26_smoke_small_custom_grid():
    data = json_cmd("smoke", "--degree", "1", "--level", "1", "--nx", "6", "--ny", "6", "--steps", "1")
    assert data["basis_dim"] == 4
    assert data["grid"] == [6, 6]
    assert data["sparse_points"] == 7
    assert data["steps"] == 1
    assert data["mass_error"] < 1e-12
    assert data["energy_final"] < data["energy_initial"]


def test_27_smoke_rectangular_custom_grid():
    data = json_cmd("smoke", "--degree", "3", "--level", "2", "--nx", "10", "--ny", "8", "--steps", "1")
    assert data["basis_dim"] == 20
    assert data["grid"] == [10, 8]
    assert data["sparse_points"] == 25
    assert data["steps"] == 1
    assert_close(data["energy_initial"], 0.25131842422056067, tol=1e-14)
    assert_close(data["energy_final"], 0.2507725007662598, tol=1e-14)


def test_28_smoke_seed_is_deterministic():
    args = ("smoke", "--degree", "2", "--level", "2", "--nx", "8", "--ny", "8", "--steps", "2", "--seed", "7")
    first = json_cmd(*args)
    second = json_cmd(*args)
    assert first == second


def test_29_missing_required_owen_argument_fails():
    result = run_cmd("owen", "--h", "1", check=False)
    assert result.returncode == 2
    assert "the following arguments are required: --a" in result.stderr


def test_30_unknown_subcommand_fails():
    result = run_cmd("does-not-exist", check=False)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr

