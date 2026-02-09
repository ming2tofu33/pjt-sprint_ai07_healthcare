"""src.dataprep 패키지 — 하위 모듈 re-export.

data_pipeline.py 등 기존 코드가 flat import 경로
(예: ``from src.dataprep.dedup import ...``)를 사용하므로,
실제 하위 패키지의 모듈을 이 수준에서 lazy하게 노출한다.
"""

from __future__ import annotations

import importlib
import types
from typing import Any


# flat 이름 → 실제 모듈 경로 매핑
_MODULE_MAP: dict[str, str] = {
    "io_utils": "src.dataprep.setup.io_utils",
    "normalize": "src.dataprep.process.normalize",
    "dedup": "src.dataprep.process.dedup",
    "split": "src.dataprep.process.split",
    "quality_audit": "src.dataprep.process.quality_audit",
    "export": "src.dataprep.output.export",
    "manifest": "src.dataprep.output.manifest",
    "data_pipeline": "src.dataprep.output.data_pipeline",
    # data_pipeline.py가 ``from src.dataprep.config import resolve_path``로 접근
    "config": "src.utils.config_loader",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        mod = importlib.import_module(_MODULE_MAP[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'src.dataprep' has no attribute {name!r}")
