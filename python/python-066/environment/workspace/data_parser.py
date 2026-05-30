
import numpy as np
from typing import List, Dict, Optional


def parse_numeric_table(lines: List[str],
                        comment_char: str = '#',
                        delimiter: Optional[str] = None) -> np.ndarray:
    data_rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(comment_char):
            continue

        if comment_char in stripped:
            stripped = stripped.split(comment_char)[0].strip()
        parts = stripped.split(delimiter) if delimiter else stripped.split()
        try:
            row = [float(p) for p in parts]
            if row:
                data_rows.append(row)
        except ValueError:
            continue
    if not data_rows:
        return np.array([])

    n_cols = len(data_rows[0])
    filtered = [r for r in data_rows if len(r) == n_cols]
    return np.array(filtered, dtype=float)


def format_well_report(well_data: List[Dict],
                       include_encoded_id: bool = False,
                       encoder=None) -> str:
    header = "# Well Monitoring Report\n"
    header += "# {:<12} {:>10} {:>10} {:>10} {:>10} {:>12}\n".format(
        "ID", "X(m)", "Y(m)", "Depth(m)", "Head(m)", "Conc(mg/L)")

    lines = [header]
    for well in well_data:
        wid = well.get("well_id", "N/A")
        if include_encoded_id and encoder is not None:
            wid = encoder(wid)
        x = well.get("x", 0.0)
        y = well.get("y", 0.0)
        depth = well.get("depth", 0.0)
        head = well.get("head", 0.0)
        conc = well.get("concentration", 0.0)
        lines.append("{:<12} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>12.6f}\n".format(
            wid, x, y, depth, head, conc))
    return "".join(lines)


def serialize_time_series(t: np.ndarray, values: np.ndarray,
                          variable_name: str = "concentration",
                          unit: str = "mg/L") -> str:
    if len(t) != len(values):
        raise ValueError("时间数组与数值数组长度不一致")
    lines = [
        f"# Time series: {variable_name}",
        f"# Unit: {unit}",
        f"# N_points: {len(t)}",
        "# t          value",
    ]
    for ti, vi in zip(t, values):
        lines.append(f"{ti:.6f}   {vi:.10e}")
    return "\n".join(lines) + "\n"


def parse_time_series(text: str) -> dict:
    lines = text.strip().split('\n')
    t_list = []
    v_list = []
    meta = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if ':' in line:
                key, val = line[1:].split(':', 1)
                meta[key.strip()] = val.strip()
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                t_list.append(float(parts[0]))
                v_list.append(float(parts[1]))
            except ValueError:
                continue
    return {
        "meta": meta,
        "t": np.array(t_list, dtype=float),
        "values": np.array(v_list, dtype=float),
    }


if __name__ == "__main__":
    text_lines = [
        "# monitoring wells",
        "MW-01  100.0  200.0  45.0",
        "MW-02  150.0  220.0  50.0",
        "",
        "# end"
    ]
    arr = parse_numeric_table(text_lines)
    assert arr.shape == (2, 4)

    ts_text = serialize_time_series(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
    parsed = parse_time_series(ts_text)
    assert len(parsed["t"]) == 3
    print("data_parser: 自测试通过")
