"""src.utils.logger — 구조화된 로깅 유틸리티.

사용:
    from src.utils.logger import get_logger
    log = get_logger("stage_0")
    log.info("처리 시작")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """포맷된 Logger를 반환한다. 동일 이름이면 기존 Logger를 재사용한다."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 파일 핸들러 (선택적)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
