"""v8 판정: docs/V8_DESIGN.md 10절 합격선을 갈래별로 적용해 runs/repeated_v8/judgement.md를 만든다.

입력: 반복 평가(evaluate_repeated.py --out runs/repeated_v8)의 run별 results/summary_repeated.json,
      하중 디딤 분석(stance_load_analysis.py --out runs/propulsion_v8)의 stance_load.json.
값은 학습 seed 3개(_s1, _s2 포함) 평균. 판정은 노이즈 끔, 노이즈 켬은 보조 기준.

    python v8_judge.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = ("", "_s1", "_s2")
CONFIGS = ("main_v6Bb", "main_v8Bb", "main_v8Db", "main_v8Eb", "main_v8Cb")
LABELS = {"main_v6Bb": "A (토크, K0) = v6Bb", "main_v8Bb": "B (PD, K0)", "main_v8Db": "D (PD, K1)",
          "main_v8Eb": "E (토크, K1)", "main_v8Cb": "C (토크, K0, σ 0.1)"}

# (항목 이름, 지표, 비교, 합격선) — 노이즈 끔
OFF_RULES = [
    ("추종 성공률", "track_rate", ">=", 0.5), ("속도", "speed", ">=", 0.40),
    ("이동 성공률", "move_rate", ">=", 0.95), ("영구 굳음", "permanent_freeze", "<=", 0.0),
    ("디딤 지속", "load_dur", ">=", 0.10), ("하중 시간 비율", "load_frac", ">=", 0.30),
    ("네 발 공중", "all_air", "<=", 0.20), ("미끄러짐", "load_slip", "<=", 0.50),
    ("관절 한계", "limit_occ", "<=", 0.20), ("넘어짐", "fell", "<=", 0.02),
]
# 노이즈 켬 보조 기준 (속도는 끔 대비 비율로 따로)
ON_RULES = [
    ("켬 넘어짐", "fell", "<=", 0.05), ("켬 이동", "move_rate", ">=", 0.90),
    ("켬 굳음", "permanent_freeze", "<=", 0.0), ("켬 디딤", "load_dur", ">=", 0.07),
]
EPS = 1e-9


def _ok(v, op, thr):
    return v >= thr - EPS if op == ">=" else v <= thr + EPS


def judge(off: dict, on: dict) -> tuple[str, dict[str, bool]]:
    """(판정, 항목별 통과). 판정: 걷는다 / 결정론적으로만 걷는다 / 서 있기만 한다 / 그 밖: <못 넘은 항목>."""
    checks = {name: _ok(off[k], op, thr) for name, k, op, thr in OFF_RULES}
    on_checks = {name: _ok(on[k], op, thr) for name, k, op, thr in ON_RULES}
    on_checks["켬 속도 유지"] = on["speed"] >= 0.7 * off["speed"] - EPS
    checks.update(on_checks)
    if off["speed"] < 0.05 and off["move_rate"] < 0.2:
        return "서 있기만 한다", checks
    failed_off = [name for name, *_ in OFF_RULES if not checks[name]]
    if failed_off:
        return "그 밖: " + ", ".join(failed_off), checks
    if not all(on_checks.values()):
        return "결정론적으로만 걷는다", checks
    return "걷는다", checks


def _mode_metrics(summaries, stance, mode, key):
    m = "deterministic" if mode == "off" else "stochastic"
    agg = [s[m] for s in summaries]
    mean = lambda f: float(np.mean([f(a) for a in agg]))
    st = stance.get(f"{key}/{mode}", {}).get("mean", {})
    return dict(
        track_rate=mean(lambda a: a["track_rate"]), speed=mean(lambda a: a["speed_3_50"]),
        move_rate=mean(lambda a: a["move_rate"]),
        permanent_freeze=mean(lambda a: a["outcomes"]["permanent_freeze"] + a["outcomes"]["initial_freeze"]),
        fell=mean(lambda a: a["outcomes"]["fell"]), all_air=mean(lambda a: a["all_feet_air_ratio"]),
        limit_occ=mean(lambda a: float(np.mean(a["joint_limit_occupancy"]))),
        load_dur=st.get("load_dur_mean", float("nan")), load_frac=st.get("load_time_frac", float("nan")),
        load_slip=st.get("load_slip_gt_0_1", float("nan")))


def _per_speed(summaries, mode):
    det = mode == "off"
    out = {}
    for name, sel in (("0.5", lambda i: i < 8), ("1.0", lambda i: i >= 8)):
        eps = [e for s in summaries for e in s["episodes"] if e["deterministic"] == det and sel(e["cmd_idx"])]
        out[name] = (float(np.mean([e["tracking"] for e in eps])), float(np.mean([e["speed_3_50"] for e in eps])))
    return out


def load(repeated: Path, run: str):
    p = Path("runs") / run / "results" / "summary_repeated.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stance", default="runs/propulsion_v8/stance_load.json")
    ap.add_argument("--out", default="runs/repeated_v8/judgement.md")
    a = ap.parse_args()
    stance = json.loads(Path(a.stance).read_text())["result"] if Path(a.stance).exists() else {}
    fmt = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    lines = ["# v8 판정 (docs/V8_DESIGN.md 10절)", "",
             "학습 seed 3개 평균, 이어 학습한 run. 판정은 노이즈 끔, 노이즈 켬은 보조 기준.", ""]
    rows = []
    for cfg in CONFIGS:
        sums = [s for s in (load(None, cfg + sfx) for sfx in SEEDS) if s]
        if not sums:
            continue
        off, on = _mode_metrics(sums, stance, "off", cfg), _mode_metrics(sums, stance, "on", cfg)
        label, checks = judge(off, on)
        rows.append((cfg, off, on, label, checks, _per_speed(sums, "off"), _per_speed(sums, "on")))
    head = "| 항목 | 합격선 | " + " | ".join(LABELS[r[0]] for r in rows) + " |"
    lines += ["## 노이즈 끔 (판정)", "", head, "|" + " --- |" * (len(rows) + 2)]
    for name, k, op, thr in OFF_RULES:
        lines.append(f"| {name} | {op} {thr:g} | " + " | ".join(
            f"{fmt(r[1][k])} {'✅' if r[4][name] else '❌'}" for r in rows) + " |")
    lines += ["", "## 노이즈 켬 (보조 기준)", "", head, "|" + " --- |" * (len(rows) + 2)]
    for name, k, op, thr in ON_RULES:
        lines.append(f"| {name} | {op} {thr:g} | " + " | ".join(
            f"{fmt(r[2][k])} {'✅' if r[4][name] else '❌'}" for r in rows) + " |")
    lines.append("| 켬 속도 유지 | ≥ 끔의 70% | " + " | ".join(
        f"{fmt(r[2]['speed'])} {'✅' if r[4]['켬 속도 유지'] else '❌'}" for r in rows) + " |")
    lines += ["", "## 판정", ""] + [f"- **{LABELS[r[0]]}:** {r[3]}" for r in rows]
    lines += ["", "## 명령 속도별 추종 성공률 / 명령 방향 속도 (끔 · 켬)", "",
              "| 갈래 | 0.5 m/s 끔 | 1.0 m/s 끔 | 0.5 m/s 켬 | 1.0 m/s 켬 |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        cell = lambda d, s: f"{d[s][0]:.2f} / {d[s][1]:.2f}"
        lines.append(f"| {LABELS[r[0]]} | {cell(r[5], '0.5')} | {cell(r[5], '1.0')} | {cell(r[6], '0.5')} | {cell(r[6], '1.0')} |")
    lines += ["", "## 학습 전 대비 (노이즈 켬: 넘어짐 / 명령 방향 속도 / 평균 |토크| N·m)", "",
              "| 갈래 | 학습 전 | 학습 후 |", "| --- | --- | --- |"]
    for cfg in CONFIGS[1:]:
        base = cfg[:-1]   # main_v8Bb → main_v8B
        pre = [s for s in (load(None, f"{base}{sfx}_init") for sfx in SEEDS) if s]
        post = [s for s in (load(None, cfg + sfx) for sfx in SEEDS) if s]
        if not pre or not post:
            continue
        f = lambda ss: (np.mean([s["stochastic"]["outcomes"]["fell"] for s in ss]),
                        np.mean([s["stochastic"]["speed_3_50"] for s in ss]),
                        np.mean([s["stochastic"].get("mean_abs_torque") or np.nan for s in ss]))
        lines.append(f"| {LABELS[cfg]} | {'%.2f / %.3f / %.1f' % f(pre)} | {'%.2f / %.3f / %.1f' % f(post)} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved →", a.out)


if __name__ == "__main__":
    main()
