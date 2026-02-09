"""Re-export: src.dataprep.process.quality_audit (lazy).

quality_audit 모듈은 cv2/numpy 등 무거운 의존성을 top-level 에서 import 하므로
실제 사용 시점(함수 호출)까지 로드를 지연시킨다.

data_pipeline.py 가 사용하는 두 함수만 래핑한다:
- run_pixel_overlap_audit
- run_aux_detector_audit
"""


def run_pixel_overlap_audit(*args, **kwargs):  # noqa: D103
    from src.dataprep.process.quality_audit import run_pixel_overlap_audit as _impl
    return _impl(*args, **kwargs)


def run_aux_detector_audit(*args, **kwargs):  # noqa: D103
    from src.dataprep.process.quality_audit import run_aux_detector_audit as _impl
    return _impl(*args, **kwargs)
