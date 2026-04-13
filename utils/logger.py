import csv
import os
from typing import Any, Dict, List


class Logger:
    """Logs scalar metrics to stdout and a CSV file."""

    def __init__(self, csv_path: str, fields: List[str]):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self._csv_path = csv_path
        self._fields = fields
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, data: Dict[str, Any], *, print_line: bool = True) -> None:
        """Write one row of metrics.

        Args:
            data:       mapping of field name → value (must match fields list)
            print_line: if True, also print a formatted line to stdout
        """
        if print_line:
            parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                     for k, v in data.items()]
            print("  ".join(parts))

        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(data)
