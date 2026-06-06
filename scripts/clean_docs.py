"""
Layer 2: Data Cleaning
----------------------
3GPP specs are Word documents zipped as .zip files. Inside each zip is a .doc/.docx.
We extract, convert to plain text, and strip the boilerplate that would pollute
our vector index (headers, footers, revision history tables, legal notices).

RAG concept: garbage in, garbage out. Noisy text produces noisy embeddings.
The cleaner your chunks, the more precise your retrieval.
"""

import os
import re
import zipfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_RAW_PATH, DATA_CLEANED_PATH

# Patterns that identify boilerplate lines to drop
BOILERPLATE_PATTERNS = [
    r"^3GPP TS \d+\.\d+ V\d+",            # version header lines
    r"^\s*Release \d+\s*$",                # "Release 17" dividers
    r"^\s*\d+\s*$",                         # lone page numbers
    r"^ETSI$",
    r"^Post(al)? address",
    r"^Copyright",
    r"^No part may be reproduced",
    r"^The copyright",
    r".*Trade Mark.*",
    r".*registered.*benefit.*members.*",
    r"^Tel\.:.*Fax:",
    r"^650 Route",
    r"^Valbonne",
    r"^Sophia Antipolis",
    r"^Internet$",
    r"^http://www\.3gpp\.org",
    r"^3GPP support office",
    r"^Postal address",
    r"^All rights reserved",
    r".*Organizational Partners.*liability.*",
    r"^\s*\[\d+\]\s+3GPP (TS|TR) \d+\.\d+", # numbered reference list entries [1] 3GPP TS...
    r"^\s*$",                               # blank lines (collapsed later)
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.MULTILINE)


def extract_text_from_zip(zip_path: str) -> str:
    """Extract and concatenate text from all .txt/.doc content inside the zip."""
    texts = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith((".txt",)):
                with z.open(name) as f:
                    texts.append(f.read().decode("utf-8", errors="replace"))
            elif name.endswith((".docx",)):
                try:
                    import docx
                    from io import BytesIO
                    with z.open(name) as f:
                        doc = docx.Document(BytesIO(f.read()))
                        blocks = []
                        for block in doc.element.body:
                            tag = block.tag.split("}")[-1]
                            if tag == "p":
                                # Regular paragraph
                                para = docx.text.paragraph.Paragraph(block, doc)
                                if para.text.strip():
                                    blocks.append(para.text)
                            elif tag == "tbl":
                                # Table — render as markdown so LLM understands structure
                                table = docx.table.Table(block, doc)
                                rows = []
                                for i, row in enumerate(table.rows):
                                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                                    rows.append("| " + " | ".join(cells) + " |")
                                    if i == 0:
                                        rows.append("|" + "|".join(["---"] * len(cells)) + "|")
                                blocks.append("\n".join(rows))
                        texts.append("\n\n".join(blocks))
                except ImportError:
                    print("  [warn] python-docx not installed — run: pip install python-docx")
    return "\n".join(texts)


def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        if _BOILERPLATE_RE.match(line):
            continue
        cleaned.append(line)
    # Collapse 3+ consecutive blank lines into 2
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    os.makedirs(DATA_CLEANED_PATH, exist_ok=True)
    zips = [f for f in os.listdir(DATA_RAW_PATH) if f.endswith(".zip")]
    if not zips:
        print(f"No zip files in {DATA_RAW_PATH}/ — run fetch_specs.py first.")
        return

    for filename in zips:
        zip_path = os.path.join(DATA_RAW_PATH, filename)
        out_name = filename.replace(".zip", ".txt")
        out_path = os.path.join(DATA_CLEANED_PATH, out_name)

        if os.path.exists(out_path):
            print(f"  [skip] {out_name} already cleaned")
            continue

        print(f"  Cleaning {filename} ...")
        raw = extract_text_from_zip(zip_path)
        if not raw:
            print(f"  [warn] No extractable text found in {filename}")
            continue
        cleaned = clean_text(raw)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"  -> {out_name} ({len(cleaned):,} chars)")

    print("\nDone. Next step: run scripts/embed.py")


if __name__ == "__main__":
    main()
