"""v1 재현(main_v1r) vs v2(main) 학습 곡선: approx_kl, 행동 std, 버틴 길이. 패널마다 축 하나.

    python plot_v1r.py                                    # → runs/v1r_compare/v1r_vs_v2.png
    python plot_v1r.py runs out.png "Noto Sans CJK KR"    # 로그 폴더, 출력 파일, 한글 글꼴
텐서보드 로그(runs/<run>/PPO_*)가 필요하다 (.gitignore로 저장소에는 없음).
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/v1r_compare/v1r_vs_v2.png")
font = sys.argv[3] if len(sys.argv) > 3 else "Noto Sans CJK KR"
plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, GRID, SURF = "#16202B", "#5A6270", "#DDD7CA", "#F5F3EE"
RUNS = [("v1 재현", "main_v1r", "#C4501B"), ("v2", "main", "#2266AA")]
PANELS = [("train/approx_kl", "approx_kl (로그 축)", True, "정상 0.01~0.03"),
          ("train/std", "행동 표준편차", False, None),
          ("rollout/ep_len_mean", "버틴 길이 (최대 1000)", False, None)]

data = {}
for label, run, _ in RUNS:
    ea = EventAccumulator(sorted(glob.glob(str(root / run / "PPO_*")))[-1])
    ea.Reload()
    data[run] = {t: ([s.step / 1e6 for s in ea.Scalars(t)], [s.value for s in ea.Scalars(t)])
                 for t, *_ in PANELS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
for ax, (tag, title, log, note) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    for label, run, color in RUNS:
        x, y = data[run][tag]
        ax.plot(x, y, color=color, lw=2, label=label)
        ax.annotate(f"{label}  {y[-1]:.3g}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11, color=INK)
    if log:
        ax.set_yscale("log")
        ax.axhspan(0.01, 0.03, color=GRID, alpha=0.7, lw=0)
        ax.text(5.1, 0.02, note, fontsize=10, color=MUTED, va="center")
    ax.set_title(title, fontsize=14, color=INK, loc="left")
    ax.set_xlabel("학습 스텝 (백만)", fontsize=11, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_xlim(0, 6.2)
axes[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, facecolor=SURF)
print("saved", out)
