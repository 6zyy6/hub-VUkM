from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import Phase


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, day: int, phase: Phase | str, event_type: str, **data: Any) -> None:
        record = {
            "day": day,
            "phase": phase.value if isinstance(phase, Phase) else phase,
            "type": event_type,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
