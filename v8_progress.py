"""run_v8.sh 진행률: 학습 run별 스텝, 전체 진행률, 예상 남은 시간.

    python v8_progress.py                 # 한 번 보기
    watch -n 30 .venv/bin/python v8_progress.py   # 30초마다 새로 보기 (Ctrl+C로 끝)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

ARMS = "BDEC"
SEEDS = ("", "_s1", "_s2")
MAIN, CONT = 5_000_000, 1_600_000
NAMES = {"B": "PD + 기존 커널", "D": "PD + 두 항 커널", "E": "토크 + 두 항 커널", "C": "토크 + 작은 노이즈"}


def steps_of(log: Path) -> int:
    """학습 로그에서 마지막 total_timesteps (이어 학습은 500만부터 세므로 그대로 둔다)."""
    if not log.exists():
        return 0
    with open(log, "rb") as f:
        f.seek(max(0, os.path.getsize(log) - 20000))
        tail = f.read().decode(errors="ignore")
    m = re.findall(r"total_timesteps\s*\|\s*(\d+)", tail)
    return int(m[-1]) if m else 0


def main():
    runs = Path("runs")
    order = [(f"main_v8{a}{s}", f"main_v8{a}b{s}", a, s) for s in SEEDS for a in ARMS]
    total, done_steps, rows = len(order) * (MAIN + CONT), 0, []
    for base, cont, arm, sfx in order:
        b_done = (runs / base / "final_model.zip").exists()
        c_done = (runs / cont / "final_model.zip").exists()
        b = MAIN if b_done else min(steps_of(runs / f"{base}.log"), MAIN)
        c = CONT if c_done else max(0, min(steps_of(runs / f"{cont}.log") - MAIN, CONT))
        done_steps += b + c
        state = "완료" if c_done else ("이어 학습 중" if b_done and c else ("이어 학습 대기" if b_done else ("학습 중" if b else "대기")))
        rows.append(f"  {arm}{sfx or '_s0':4s} {NAMES[arm]:14s} {state:8s} {100 * (b + c) / (MAIN + CONT):5.1f}%")
    log = runs / "v8_all.log"
    print(f"v8 학습 진행률: {100 * done_steps / total:5.1f}%  ({done_steps / 1e6:.1f}M / {total / 1e6:.1f}M 스텝)")
    if log.exists():
        lines = log.read_text().splitlines()
        start = next((l for l in lines if "start (" in l), None)
        if start and done_steps:
            t0 = time.mktime(time.strptime(time.strftime("%Y-") + start[1:15], "%Y-%m-%d %H:%M:%S"))
            el = time.time() - t0
            left = el * (total - done_steps) / done_steps
            print(f"경과 {el / 3600:.1f}시간, 남은 학습 약 {left / 3600:.1f}시간 (+ 평가 약 20분)")
        print("마지막 기록:", lines[-1] if lines else "–")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
