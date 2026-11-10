"""여러 seed의 평가 결과(summary.json)를 평균 ± 표준편차와 방향별 속도로 정리.

    python seed_stats.py                       # v5, v5b × seed 0, 1, 2
    python seed_stats.py v5=main_v5,main_v5_s1 v5b=main_v5b,main_v5b_s1
"""
import json
import math
import statistics as st
import sys

DEFAULT = {"v5": ["main_v5", "main_v5_s1", "main_v5_s2"],
           "v5b": ["main_v5b", "main_v5b_s1", "main_v5b_s2"]}

METRICS = [
    ("노이즈 끔 오차", lambda s: s["deterministic"]["mean_tracking_error_mps"]),
    ("노이즈 켬 오차", lambda s: s["stochastic"]["mean_tracking_error_mps"]),
    ("노이즈 끔 속도", lambda s: s["deterministic"]["mean_along_cmd_speed"]),
    ("노이즈 켬 속도", lambda s: s["stochastic"]["mean_along_cmd_speed"]),
    ("켬 − 끔 속도", lambda s: s["stochastic"]["mean_along_cmd_speed"]
                              - s["deterministic"]["mean_along_cmd_speed"]),
    ("끔에서 움직인 명령 (> 0.05)", lambda s: sum(r["along_cmd_speed"] > 0.05 for r in s["tracking"])),
    ("넘어짐 (켬)", lambda s: s["stochastic"]["falls"]),
    ("포화 비율", lambda s: s["action_diagnostics"]["saturation_ratio"]),
    ("관절 각속도", lambda s: s["action_diagnostics"]["joint_velocity"]),
    ("제어 노력 e=0", lambda s: s["energy"]["effort"][0]),
    ("제어 노력 e=1", lambda s: s["energy"]["effort"][-1]),
]


def main():
    groups = dict(a.split("=", 1) for a in sys.argv[1:]) or None
    groups = {k: v.split(",") for k, v in groups.items()} if groups else DEFAULT
    S = {k: [json.load(open(f"runs/{r}/results/summary.json")) for r in v] for k, v in groups.items()}

    for name, f in METRICS:
        for k, ss in S.items():
            xs = [f(s) for s in ss]
            sd = st.stdev(xs) if len(xs) > 1 else 0.0
            print(f"{name:24s} {k:5s} {st.mean(xs):.3f} ± {sd:.3f}  ({', '.join(f'{x:.3f}' for x in xs)})")

    print("\n방향별 명령 방향 속도, 노이즈 끔 (각 seed 순서대로). 0° = +x, 90° = +y")
    first = next(iter(S.values()))[0]["tracking"]
    for i, r in enumerate(first):
        cx, cy = r["cmd"]
        ang = round(math.degrees(math.atan2(cy, cx))) % 360
        line = f"{math.hypot(cx, cy):.1f} m/s {ang:3d}° "
        for k, ss in S.items():
            line += f" {k}: " + " ".join(f"{s['tracking'][i]['along_cmd_speed']:6.3f}" for s in ss)
        print(line)


if __name__ == "__main__":
    main()
