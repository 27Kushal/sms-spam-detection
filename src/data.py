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
