import csv
from pathlib import Path
from datetime import datetime

class EventLogger:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists():
            with self.path.open("w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp","track_id","color","x","y","event"])
    def log(self,track_id,color,center,event):
        with self.path.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="milliseconds"),
                track_id,color,center[0],center[1],event])
