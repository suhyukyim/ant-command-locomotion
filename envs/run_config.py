"""run 폴더의 config.json ↔ CommandAnt 환경 인자.

학습한 run을 평가·분석할 때 행동 방식(토크/PD)·PD 이득·추종 커널·보상 가중치를 그 run의 설정에서 읽어
환경을 만든다. 스크립트마다 따로 넘기면 하나만 빠뜨려도 PD로 학습한 정책을 토크 환경에서 굴리는 식으로
조용히 틀린 결과가 나오기 때문이다.

옛 run(v8 이전)의 config.json에는 action_mode 등이 없으므로 CommandAnt 기본값(토크, 한 항 커널)이 된다.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from .command_ant import CommandAnt

ENV_KEYS = ("tracking_sigma", "tracking_weight", "alive_weight", "limit_weight", "limit_soft", "vz_weight",
            "action_mode", "pd_kp", "pd_kd", "tracking_kernel",
            "stance_weight", "slip_weight", "stance_min", "stance_cap")
# 이어 학습 때 원래 run과 달라지면 안 되는 것 (정책이 배운 행동의 뜻이나 보상 모양이 바뀐다)
RESUME_LOCKED = ("action_mode", "pd_kp", "pd_kd", "tracking_kernel", "stance_weight", "slip_weight")

_DEFAULTS = {k: p.default for k, p in inspect.signature(CommandAnt.__init__).parameters.items() if k in ENV_KEYS}


def env_kwargs_from_config(cfg: dict) -> dict:
    """config.json 내용 → make_command_ant 인자. 없는 키는 CommandAnt 기본값."""
    return {k: cfg.get(k, _DEFAULTS[k]) if cfg.get(k) is not None else _DEFAULTS[k] for k in ENV_KEYS}


def load_env_kwargs(run_dir: str | Path) -> dict:
    return env_kwargs_from_config(json.loads((Path(run_dir) / "config.json").read_text()))


def check_resume_compatible(old: dict, new: dict) -> None:
    """이어 학습의 환경 설정이 원래 run과 같은지. 다르면 ValueError (키 이름 포함)."""
    diff = [f"{k}: {old.get(k)!r} → {new.get(k)!r}" for k in RESUME_LOCKED if old.get(k) != new.get(k)]
    if diff:
        raise ValueError("이어 학습 설정이 원래 run과 다르다: " + ", ".join(diff))
