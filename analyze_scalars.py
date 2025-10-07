import json
from pathlib import Path
import statistics
import sys

DEFAULT_PATHS = [
    Path('runs/default/scalars.json'),
    Path('scalars.json'),
]

if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    log_path = next((p for p in DEFAULT_PATHS if p.exists()), None)

if log_path is None:
    raise SystemExit('log file not found. pass explicit path as argument.')

data = json.loads(Path(log_path).read_text())


def to_series(entries):
    return [(int(step), float(val)) for step, val in entries]


series = {k: to_series(v) for k, v in data.items()}


def summarize_metric(name):
    points = series.get(name, [])
    if not points:
        return None
    best = max(points, key=lambda x: x[1])
    last = points[-1]
    trend = 'flat'
    if len(points) >= 2:
        if points[-1][1] > points[-2][1]:
            trend = 'up'
        elif points[-1][1] < points[-2][1]:
            trend = 'down'
    return dict(
        name=name,
        count=len(points),
        best_step=best[0],
        best_val=best[1],
        last_step=last[0],
        last_val=last[1],
        trend=trend,
    )


summary = []
for key in sorted(series):
    info = summarize_metric(key)
    if info:
        summary.append(info)

print(f"Loaded {log_path}")
print('=== metric summary ===')
for info in summary:
    print(
        f"{info['name']}: {info['count']} pts | best={info['best_val']:.3f} @ upd {info['best_step']} | "
        f"last={info['last_val']:.3f} @ upd {info['last_step']} | trend={info['trend']}"
    )

# extra diagnostics for success collapse
success = series.get('eval/success', [])
if success:
    best_step, best_val = max(success, key=lambda x: x[1])
    zeros_after_best = [step for step, val in success if step > best_step and val == 0.0]
    if zeros_after_best:
        first_zero = zeros_after_best[0]
        print(
            f"\nEval success drops to 0 after best result at update {first_zero} "
            f"(peak success {best_val:.2f} @ upd {best_step})."
        )

train_return = series.get('train/ep_return', [])
train_success = series.get('train/success', [])
if train_return and train_success:
    tail_returns = [val for _, val in train_return[-20:]]
    tail_success = [val for _, val in train_success[-20:]]
    mean_return = statistics.mean(tail_returns)
    mean_success = statistics.mean(tail_success)
    print(
        f"\nLast 20 updates: mean train return={mean_return:.2f}, "
        f"mean train success={mean_success:.2f}"
    )

ckpt = Path('ckpts/best.pt')
if ckpt.exists():
    size_kib = ckpt.stat().st_size / 1024
    print(f"\nBest checkpoint present at {ckpt} (size {size_kib:.1f} KiB).")
else:
    print("\nBest checkpoint not found. Save ckpts/best.pt during training for reuse.")
