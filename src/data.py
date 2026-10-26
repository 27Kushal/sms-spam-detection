"""Load the official SMS Spam Collection without CSV quoting assumptions."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"


def download_dataset(path: Path) -> Path:
    """Download the official file if absent; verify cached or downloaded bytes."""
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                content = archive.read("SMSSpamCollection")
    if sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dataset checksum differs from the verified UCI source.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Split each physical record at its first tab, preserving SMS text."""
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if "\t" not in line:
            raise ValueError(f"Source line {line_number} has no label separator.")
        label, message = line.split("\t", 1)
        records.append({"source_line": line_number, "label": label, "message": message})
    return pd.DataFrame(records, columns=["source_line", "label", "message"])


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove missing/blank records and exact duplicate SMS messages.

    Preserve case, punctuation and whitespace in nonblank SMS text. Detect
    conflicting labels before deduplication rather than choosing one silently.
    """
    required = {"source_line", "label", "message"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    missing = raw[["label", "message"]].isna()
    blank = raw[["label", "message"]].apply(lambda col: col.fillna("").str.strip().eq(""))
    invalid = (missing | blank).any(axis=1)
    valid = raw.loc[~invalid].copy()
    unknown = set(valid["label"]) - {"spam", "ham"}
    if unknown:
        raise ValueError(f"Unexpected labels: {sorted(unknown)}")
    conflicts = valid.groupby("message")["label"].nunique().gt(1)
    if conflicts.any():
        raise ValueError("Identical messages have conflicting labels; inspect the source.")
    duplicate_count = int(valid.duplicated(subset="message").sum())
    clean = valid.drop_duplicates(subset="message", keep="first").copy()
    clean["target"] = clean["label"].map({"ham": 0, "spam": 1}).astype("int64")
    clean = clean.reset_index(drop=True)
    summary = {
        "raw_records": int(len(raw)),
        "missing_values": {key: int(value) for key, value in missing.sum().items()},
        "blank_non_null_values": {
            key: int(value) for key, value in (blank & ~missing).sum().items()
        },
        "removed_missing_or_blank_records": int(invalid.sum()),
        "removed_duplicate_messages": duplicate_count,
        "clean_records": int(len(clean)),
        "raw_class_counts": {key: int(value) for key, value in raw["label"].value_counts().items()},
        "clean_class_counts": {key: int(value) for key, value in clean["label"].value_counts().items()},
        "label_encoding": {"ham": 0, "spam": 1},
        "duplicate_rule": "Exact message text; retain the first occurrence",
    }
    return clean, summary
