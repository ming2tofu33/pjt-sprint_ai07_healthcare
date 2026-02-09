"""Re-export: src.dataprep.setup.io_utils → process/io_utils.

quality_audit.py 등 process 하위 모듈에서 ``from .io_utils import ...`` 패턴을
사용할 수 있도록 setup 패키지의 io_utils 를 중계한다.
"""
from src.dataprep.setup.io_utils import *  # noqa: F401,F403
from src.dataprep.setup.io_utils import (  # noqa: F401
    scan_json_files,
    scan_image_files,
    read_json_with_fallbacks,
    parse_one_json,
)
