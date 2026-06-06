"""
Layer 1: Data Ingestion
-----------------------
Downloads 3GPP specification zips from the public 3gpp.org FTP server.

3GPP organises specs by series:
  - 36.xxx = LTE (4G)
  - 38.xxx = NR (5G)

We scrape each spec's directory listing to find the latest zip automatically,
so version strings never go stale.
"""

import os
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_RAW_PATH

SPECS_TO_FETCH = [
    {
        "series": "36",
        "spec": "36300",
        "description": "LTE Overall Description (E-UTRA/E-UTRAN)",
    },
    {
        "series": "38",
        "spec": "38300",
        "description": "5G NR Overall Description",
    },
]

# Folder name on 3GPP FTP uses a dot: 36.300, 38.300, etc.
DIR_URL = "https://www.3gpp.org/ftp/Specs/archive/{series}_series/{spec_dot}/"


def spec_to_dot(spec: str) -> str:
    """'36300' -> '36.300'"""
    return spec[:2] + "." + spec[2:]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_latest_zip_url(series: str, spec: str) -> tuple[str, str] | tuple[None, None]:
    """Scrape the spec directory and return (url, filename) for the latest zip.

    The 3GPP site is ASP.NET — we use a Session so the server sets a cookie on
    the first request and returns a proper file listing on subsequent requests.
    File links have class="file" and use full absolute URLs as hrefs.
    """
    spec_dot = spec_to_dot(spec)
    dir_url = DIR_URL.format(series=series, spec_dot=spec_dot)

    print(f"  Checking directory: {dir_url}")
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # First hit establishes the ASP.NET session cookie
        session.get("https://www.3gpp.org/ftp/Specs/archive/", timeout=30)
        # Second hit fetches the actual directory with the cookie in place
        r = session.get(dir_url, timeout=30)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"  [error] Could not list directory: {e}")
        return None, None

    soup = BeautifulSoup(r.text, "html.parser")

    # Target <a class="file"> — the exact tag the 3GPP site uses for file links
    zip_links = sorted(
        a["href"] for a in soup.find_all("a", class_="file", href=True)
        if a["href"].endswith(".zip")
    )

    if not zip_links:
        # Fallback: try any href ending in .zip in case markup changes
        all_links = [a["href"] for a in soup.find_all("a", href=True)]
        zip_links = sorted(h for h in all_links if h.endswith(".zip"))

    if not zip_links:
        print(f"  [debug] Still no .zip links. First 10 hrefs on page:")
        all_links = [a["href"] for a in soup.find_all("a", href=True)]
        for link in all_links[:10]:
            print(f"    {link}")
        return None, None

    latest_href = zip_links[-1]
    latest_url = urljoin(dir_url, latest_href)
    latest_filename = latest_url.split("/")[-1]

    print(f"  Latest version: {latest_filename}")
    return latest_url, latest_filename


def fetch_spec(spec: dict, dest_dir: str) -> str | None:
    url, filename = get_latest_zip_url(spec["series"], spec["spec"])
    if not url:
        return None

    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename} already downloaded")
        return dest

    print(f"  Downloading {spec['description']} ...")
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        r = session.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return dest
    except requests.HTTPError as e:
        print(f"  [error] Could not download {url}: {e}")
        return None


def main():
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    print(f"Downloading specs to {DATA_RAW_PATH}/\n")
    for spec in SPECS_TO_FETCH:
        print(f"\n--- {spec['description']} ---")
        fetch_spec(spec, DATA_RAW_PATH)
    print("\nDone. Next step: run scripts/clean_docs.py")


if __name__ == "__main__":
    main()
