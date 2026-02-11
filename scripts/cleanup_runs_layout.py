"""runs/<run_name> 레이아웃을 컴팩트 구조로 정리하는 보조 스크립트.

기본 정리 대상:
- run 루트의 학습 시각화/플롯 파일 -> run/train/
- run/submission_debug/ -> run/submit/debug/

사용 예시:
- python scripts/cleanup_runs_layout.py --run-name y11m_ext_balanced_v1 --dry-run
- python scripts/cleanup_runs_layout.py --all
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TRAIN_ARTIFACT_PATTERNS = [
    "args.yaml",
    "results.csv",
    "results.png",
    "labels.jpg",
    "Box*.png",
    "confusion_matrix*.png",
    "train_batch*.jpg",
    "val_batch*_labels.jpg",
    "val_batch*_pred.jpg",
]


def _iter_target_runs(runs_dir: Path, run_name: str | None, use_all: bool) -> list[Path]:
    if run_name and use_all:
        raise ValueError("--run-name 과 --all 을 동시에 사용할 수 없습니다.")
    if not run_name and not use_all:
        raise ValueError("--run-name 또는 --all 중 하나는 반드시 지정해야 합니다.")

    if run_name:
        run_dir = runs_dir / run_name
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"run 디렉터리를 찾을 수 없습니다: {run_dir}")
        return [run_dir]

    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def _move_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if src.resolve() == dst.resolve():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[SKIP] 이미 존재: {dst}")
        return False

    if dry_run:
        print(f"[DRY] move {src} -> {dst}")
        return True

    shutil.move(str(src), str(dst))
    print(f"[MOVE] {src} -> {dst}")
    return True


def _copy_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] copy {src} -> {dst}")
        return True

    shutil.copy2(str(src), str(dst))
    print(f"[COPY] {src} -> {dst}")
    return True


def _move_submission_debug(run_dir: Path, dry_run: bool) -> int:
    src_dir = run_dir / "submission_debug"
    dst_dir = run_dir / "submit" / "debug"

    if not src_dir.exists() or not src_dir.is_dir():
        return 0

    moved = 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if _move_file(src, dst, dry_run):
            moved += 1

    if dry_run:
        print(f"[DRY] remove_dir {src_dir}")
    else:
        try:
            shutil.rmtree(src_dir)
            print(f"[RM] {src_dir}")
        except OSError:
            pass

    return moved


def _move_train_artifacts(run_dir: Path, dry_run: bool) -> int:
    train_dir = run_dir / "train"
    moved = 0

    for pattern in TRAIN_ARTIFACT_PATTERNS:
        for src in sorted(run_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = train_dir / src.name
            if _move_file(src, dst, dry_run):
                moved += 1

    # results.csv는 루트 shortcut을 유지한다.
    train_results_csv = train_dir / "results.csv"
    root_results_csv = run_dir / "results.csv"
    if train_results_csv.exists() and not root_results_csv.exists():
        if _copy_file(train_results_csv, root_results_csv, dry_run):
            moved += 1

    return moved


def cleanup_one_run(run_dir: Path, dry_run: bool) -> None:
    print(f"\n=== cleanup: {run_dir} ===")
    moved_train = _move_train_artifacts(run_dir, dry_run)
    moved_submit = _move_submission_debug(run_dir, dry_run)
    print(f"[DONE] train_moves={moved_train}, submit_moves={moved_submit}")


def main() -> None:
    parser = argparse.ArgumentParser(description="runs 레이아웃 정리 스크립트")
    parser.add_argument("--runs-dir", default="runs", help="runs 루트 디렉터리")
    parser.add_argument("--run-name", default=None, help="정리할 단일 run 이름")
    parser.add_argument("--all", action="store_true", help="runs 아래 모든 run 정리")
    parser.add_argument("--dry-run", action="store_true", help="실제 변경 없이 계획만 출력")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise FileNotFoundError(f"runs 디렉터리를 찾을 수 없습니다: {runs_dir}")

    targets = _iter_target_runs(runs_dir, args.run_name, args.all)
    if not targets:
        print("정리할 run이 없습니다.")
        return

    for run_dir in targets:
        cleanup_one_run(run_dir, args.dry_run)


if __name__ == "__main__":
    main()
