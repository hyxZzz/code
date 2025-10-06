# src/utils/logger.py
# ---------------------------------------
import json, os, time
from collections import defaultdict

class Logger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.scalars = defaultdict(list)
        self.t0 = time.time()

    def log_scalar(self, key: str, val: float, step: int):
        self.scalars[key].append((int(step), float(val)))

    def flush(self):
        path = os.path.join(self.out_dir, "scalars.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.scalars.items()}, f, ensure_ascii=False, indent=2)
