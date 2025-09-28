from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta | {"timestamp": time.time()}, f, indent=2)
