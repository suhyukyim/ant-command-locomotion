"""v8 판정 (docs/V8_DESIGN.md 10절 합격선) 시험."""

from v8_judge import judge

# 노이즈 끔: 모든 항목이 합격선을 겨우 넘음 (경계값 포함)
OFF = dict(track_rate=0.5, speed=0.40, move_rate=0.95, permanent_freeze=0.0, load_dur=0.10, load_frac=0.30,
           all_air=0.20, load_slip=0.50, limit_occ=0.20, fell=0.02)
# 노이즈 켬: 보조 기준을 겨우 넘음 (속도는 끔의 70%)
ON = dict(track_rate=0.3, speed=0.28, move_rate=0.90, permanent_freeze=0.0, load_dur=0.07, load_frac=0.2,
          all_air=0.3, load_slip=0.6, limit_occ=0.3, fell=0.05)


def test_walks_at_boundaries():
    label, checks = judge(OFF, ON)
    assert label == "걷는다"
    assert all(checks.values())


def test_deterministic_only():
    label, checks = judge(OFF, dict(ON, fell=0.08))
    assert label == "결정론적으로만 걷는다"
    assert checks["켬 넘어짐"] is False


def test_standing_only():
    label, _ = judge(dict(OFF, speed=0.03, move_rate=0.1, track_rate=0.0), ON)
    assert label == "서 있기만 한다"


def test_other_names_failed_items():
    label, checks = judge(dict(OFF, load_slip=0.6), ON)
    assert label.startswith("그 밖")
    assert "미끄러짐" in label and checks["미끄러짐"] is False


def test_off_fall_threshold_is_stricter():
    label, checks = judge(dict(OFF, fell=0.03), ON)
    assert checks["넘어짐"] is False and label.startswith("그 밖")
