from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Settings:
    polygon_ref: list
    stamp_roi_ref: list
    count_line_ref: list
    reference_width: int = 576
    reference_height: int = 1024
    min_area: int = 70
    max_area: int = 12000
    min_circularity: float = 0.18
    max_match_distance: int = 65
    track_max_missed: int = 12
    count_direction: str = "any"
    output_dir: str = "output"
    save_video: bool = True
    show_window: bool = True
    class_thresholds: dict | None = None
    piston_motion_threshold: float = 8.0

def load_settings(path="config/settings.json"):
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    return Settings(**raw)
