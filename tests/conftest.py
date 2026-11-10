"""pytest 설정: 저장소 루트를 import 경로에 넣고 Gymnasium 경고를 끈다."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning)
