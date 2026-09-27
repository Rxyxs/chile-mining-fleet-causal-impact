"""Downloads the two public predictive-maintenance datasets the semi-synthetic
benchmark (`semi_synthetic_dgp.py`) draws its real covariates from, straight
from the UCI Machine Learning Repository, into `data/raw/`:

- APS Failure at Scania Trucks (UCI #421): 60,000 heavy trucks in everyday
  operation, 170 anonymized operational counters/histograms, real missingness,
  real heavy tails, real collinearity. The primary "real data" source.
- AI4I 2020 Predictive Maintenance (UCI #601): 10,000 machine records with
  interpretable columns (tool wear, process temperature, torque...). Note that
  its own author describes it as a *synthetic* dataset modeled on real
  maintenance data -- it is kept as a second, interpretable source, not as
  evidence about real-world sensor distributions.

Both are regenerable and git-ignored (`data/raw/*.csv`).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = {
    "scania_aps": {
        "url": "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip",
        "files": ["aps_failure_training_set.csv", "aps_failure_description.txt"],
    },
    "ai4i_2020": {
        "url": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "files": ["ai4i2020.csv"],
    },
}


def download_dataset(name: str, raw_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    spec = DATASETS[name]
    targets = [raw_dir / f for f in spec["files"]]
    if not force and all(t.exists() for t in targets):
        return targets

    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec["url"], timeout=600) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in spec["files"]:
            (raw_dir / member).write_bytes(archive.read(member))
    return targets


def main() -> None:
    for name in DATASETS:
        paths = download_dataset(name)
        print(f"{name}: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
