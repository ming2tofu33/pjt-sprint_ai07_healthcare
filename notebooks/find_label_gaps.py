import os
import glob
import csv
import shutil

# =========================
# ✅ 너 환경에 맞게 여기만 수정
# =========================
IMG_DIR = r"C:\DLP\sprint_ai_project1_data\train_images"          # train 이미지 폴더 (실제 경로로!)
LBL_DIR = r"C:\DLP\sprint_ai_project1_data\train_labels_merged"   # 병합된 라벨 txt 폴더
OUT_DIR = r"C:\DLP\pjt-sprint_ai07_healthcare\needs_review"       # 결과 복사 폴더 (자동 생성)

# 이미지 확장자들 (필요하면 더 추가)
IMG_EXTS = (".jpg", ".jpeg", ".png")

# =========================
# 유틸 함수
# =========================
def basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def list_images(img_dir: str):
    paths = []
    for ext in IMG_EXTS:
        # 하위 폴더까지 찾고 싶으면 **/* 로 바꿔도 됨
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return paths

def count_expected_from_filename(bn: str) -> int:
    """
    파일명 규칙:
      K-003544-004543-012247-016548_0_2_0_2_90_000_200
    여기서 '_' 앞부분만 떼면:
      K-003544-004543-012247-016548
    '-'로 split하면:
      ['K','003544','004543','012247','016548'] -> 코드 개수 = 4
    """
    combo = bn.split("_")[0]
    parts = combo.split("-")
    # 맨 앞 'K' 제외
    if len(parts) >= 2 and parts[0] == "K":
        return len(parts) - 1
    # 예외: 형식이 다르면 안전하게 0
    return 0

def count_label_lines(txt_path: str) -> int:
    if not os.path.exists(txt_path):
        return -1  # 라벨 파일 자체가 없음
    cnt = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                cnt += 1
    return cnt

def ensure_dirs(out_dir: str):
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    return img_out, lbl_out

# =========================
# 메인
# =========================
def main():
    images = list_images(IMG_DIR)
    if not images:
        print("❌ IMG_DIR에서 이미지를 못 찾았어. 경로를 확인해줘:", IMG_DIR)
        return

    img_out, lbl_out = ensure_dirs(OUT_DIR)
    report_path = os.path.join(OUT_DIR, "report.csv")

    suspicious = []
    total = 0
    missing_label_file = 0
    mismatch_count = 0

    for img_path in images:
        total += 1
        bn = basename_noext(img_path)
        expected = count_expected_from_filename(bn)

        lbl_path = os.path.join(LBL_DIR, bn + ".txt")
        actual = count_label_lines(lbl_path)

        # 라벨 파일이 없으면 무조건 의심
        if actual == -1:
            missing_label_file += 1
            reason = "LABEL_FILE_MISSING"
            suspicious.append((bn, expected, actual, reason, img_path, lbl_path))
            continue

        # 파일명에서 기대되는 코드 수와 bbox 줄 수가 다르면 의심
        if expected != 0 and actual != expected:
            mismatch_count += 1
            if actual < expected:
                reason = f"MISSING_BOXES(expected {expected}, got {actual})"
            else:
                reason = f"EXTRA_BOXES(expected {expected}, got {actual})"
            suspicious.append((bn, expected, actual, reason, img_path, lbl_path))

    # 복사 + 리포트 저장
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["basename", "expected_codes", "label_lines", "reason", "image_path", "label_path"])
        for bn, expected, actual, reason, img_path, lbl_path in suspicious:
            w.writerow([bn, expected, actual, reason, img_path, lbl_path])

            # 이미지 복사
            shutil.copy2(img_path, os.path.join(img_out, os.path.basename(img_path)))
            # 라벨 복사
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(lbl_out, os.path.basename(lbl_path)))

    print("=== 결과 요약 ===")
    print("전체 이미지 수:", total)
    print("라벨 파일 자체가 없는 이미지:", missing_label_file)
    print("expected_codes != label_lines 인 이미지:", mismatch_count)
    print("총 의심 이미지:", len(suspicious))
    print("복사 위치:", OUT_DIR)
    print("리포트:", report_path)

    # 상위 몇 개만 화면에도 보여주기
    if suspicious:
        print("\n[샘플 10개]")
        for row in suspicious[:10]:
            bn, expected, actual, reason, *_ = row
            print(f"- {bn}: expected={expected}, labels={actual} => {reason}")

if __name__ == "__main__":
    main()
