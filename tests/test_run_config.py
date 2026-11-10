"""run 설정(config.json) → 환경 인자 전달, 이어 학습 검사, 옛 run 재현 시험."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from envs import make_command_ant
from envs.run_config import check_resume_compatible, env_kwargs_from_config, load_env_kwargs

ROOT = Path(__file__).resolve().parent.parent


def test_old_config_defaults_to_torque():
    kw = load_env_kwargs(ROOT / "runs/main_v6Bb")
    assert kw["action_mode"] == "torque"
    assert kw["tracking_kernel"] == "single"
    assert kw["tracking_sigma"] == 1.0
    assert kw["limit_weight"] == 1.5
    assert make_command_ant(**kw).action_mode == "torque"


def test_unknown_config_keys_are_ignored():
    kw = env_kwargs_from_config({"timesteps": 5, "seed": 1, "squash": True, "tracking_weight": 5.0})
    assert "timesteps" not in kw and "squash" not in kw
    assert kw["tracking_weight"] == 5.0


@pytest.mark.parametrize("key,a,b", [
    ("action_mode", "torque", "pd"),
    ("tracking_kernel", "single", "two_term"),
    ("pd_kp", 30.0, 60.0),
    ("pd_kd", 10.0, 15.0),
])
def test_resume_mismatch_raises(key, a, b):
    base = env_kwargs_from_config({"action_mode": "pd", "pd_kp": 30.0, "pd_kd": 10.0})
    old, new = dict(base, **{key: a}), dict(base, **{key: b})
    with pytest.raises(ValueError, match=key):
        check_resume_compatible(old, new)
    check_resume_compatible(old, dict(old))


@pytest.fixture(scope="module")
def tiny_pd_run():
    name = "_test_v8"
    for d in (ROOT / "runs" / name, ROOT / "runs" / f"{name}_init"):
        shutil.rmtree(d, ignore_errors=True)
    cmd = [sys.executable, "train.py", "--action-mode", "pd", "--pd-kp", "60", "--pd-kd", "15",
           "--tracking-kernel", "two_term", "--squash", "--timesteps", "2048", "--n-envs", "1",
           "--save-init", "--log-std-init", "-2.302585", "--run-name", name]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    yield ROOT / "runs" / name
    for d in (ROOT / "runs" / name, ROOT / "runs" / f"{name}_init"):
        shutil.rmtree(d, ignore_errors=True)


def test_policy_env_kwargs_roundtrip(tiny_pd_run):
    from policy_utils import Policy
    pol = Policy(tiny_pd_run)
    assert pol.env_kwargs["action_mode"] == "pd"
    assert pol.env_kwargs["tracking_kernel"] == "two_term"
    init = tiny_pd_run.parent / f"{tiny_pd_run.name}_init"
    assert (init / "final_model.zip").exists() and (init / "vecnormalize.pkl").exists()
    assert json.loads((init / "config.json").read_text())["action_mode"] == "pd"


def test_pd_run_builds_pd_env(tiny_pd_run):
    from policy_utils import Policy
    env = make_command_ant(**Policy(tiny_pd_run).env_kwargs)
    assert env.action_mode == "pd" and env.pd_kp == 60.0


def test_log_std_init(tiny_pd_run):
    from policy_utils import Policy
    init = Policy(tiny_pd_run.parent / f"{tiny_pd_run.name}_init")
    log_std = init.model.policy.log_std.detach().numpy()
    assert abs(log_std.mean() - math.log(0.1)) < 0.05


def test_resume_with_other_action_mode_fails(tiny_pd_run):
    cmd = [sys.executable, "train.py", "--resume", str(tiny_pd_run), "--timesteps", "1024", "--n-envs", "1",
           "--run-name", "_test_v8_bad"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    shutil.rmtree(ROOT / "runs/_test_v8_bad", ignore_errors=True)
    assert r.returncode != 0 and "action_mode" in r.stderr


def test_v6Bb_eval_reproduces():
    """커밋된 반복 평가 기록의 에피소드 몇 개를 다시 돌려 같은지 본다 (옛 run은 토크로 재현돼야 한다)."""
    import evaluate_repeated as er
    rec = json.loads((ROOT / "runs/main_v6Bb/results/summary_repeated.json").read_text())["episodes"]
    picks = [e for e in rec if e["reset_seed"] == 0 and e["cmd_idx"] in (0, 12)]
    picks = [e for e in picks if e["deterministic"]] + [e for e in picks if not e["deterministic"]][:1]
    assert len(picks) == 3
    for e in picks:
        got = er.episode("runs/main_v6Bb", e["cmd"], e["reset_seed"], e["deterministic"], 1000)
        assert got["fell"] == e["fell"] and got["outcome"] == e["outcome"]
        assert abs(got["speed_3_50"] - e["speed_3_50"]) < 1e-9, (e["cmd_idx"], e["deterministic"])
