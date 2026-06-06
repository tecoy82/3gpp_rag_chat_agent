"""
Layer 1: Data Ingestion
-----------------------
Downloads 3GPP specification PDFs from the public 3gpp.org FTP server.

3GPP organises specs by series:
  - 36.xxx = LTE (4G)
  - 38.xxx = NR (5G)

We start small: two overview specs so the pipeline runs fast locally.
Add more entries to SPECS_TO_FETCH once you've validated end-to-end.
"""

import os
import requests
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_RAW_PATH

# (spec_number, version, description)
# Find the latest version zip at: https://www.3gpp.org/ftp/Specs/archive/
SPECS_TO_FETCH = [
    {
        "series": "36",
        "spec": "36300",
        "version": "m20",   # update to latest if needed
        "description": "LTE Overall Description (E-UTRA/E-UTRAN)",
    },
    {
        "series": "38",
        "spec": "38300",
        "version": "h10",
        "description": "5G NR Overall Description",
    },
]

BASE_URL = "https://www.3gpp.org/ftp/Specs/archive/{series}_series/{spec}/{spec}-{version}.zip"


def fetch_spec(spec: dict, dest_dir: str) -> str | None:
    url = BASE_URL.format(**spec)
    filename = f"{spec['spec']}-{spec['version']}.zip"
    dest = os.path.join(dest_dir, filename)

    if os.path.exists(dest):
        print(f"  [skip] {filename} already downloaded")
        return dest

    print(f"  Fetching {spec['description']} ({filename}) ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return dest
    except requests.HTTPError as e:
        print(f"  [error] Could not fetch {url}: {e}")
        return None


def main():
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    print(f"Downloading specs to {DATA_RAW_PATH}/\n")
    for spec in SPECS_TO_FETCH:
        fetch_spec(spec, DATA_RAW_PATH)
    print("\nDone. Next step: run scripts/clean_docs.py")


if __name__ == "__main__":
    main()
