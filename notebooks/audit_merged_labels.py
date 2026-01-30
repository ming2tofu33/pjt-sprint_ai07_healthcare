import os, glob, json, random
from collections import Counter, defaultdict

# ✅ 너 환경에 맞게 여기만 수정
IMG_DIR = r"C:\DLP\sprint_ai_project1_data\train_images"         # train 이미지 폴더(실제 경로로!)
LBL_DIR = r"C:\DLP\sprint_ai_project1_data\train_labels_merged"  # 병합된 txt 폴더
CLASS_MAP = os.path.join(LBL_DIR, "_class_map.json")

IMG_EXTS = (".jpg", ".jpeg", ".png")

def list_images(img_dir):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(img_dir, f"*{ext}"))
    return paths

def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_class_count():
    if os.path.exists(CLASS_MAP):
        with open(CLASS_MAP, "r", encoding="utf-8") as f:
            m = json.load(f)
        return len(m["idx2name"])
    return None

def parse_label_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                rows.append(("BAD_FORMAT", ln))
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                rows.append((cls, xc, yc, w, h))
            except:
                rows.append(("BAD_PARSE", ln))
    return rows

def main():
    imgs = list_images(IMG_DIR)
    lbls = glob.glob(os.path.join(LBL_DIR, "*.txt"))
    imgs_bn = {basename_noext(p): p for p in imgs}
    lbls_bn = {basename_noext(p): p for p in lbls if not os.path.basename(p).startswith("_")}

    print("=== COUNT ===")
    print("images:", len(imgs_bn))
    print("labels:", len(lbls_bn))

    missing_lbl = sorted(set(imgs_bn) - set(lbls_bn))
    missing_img = sorted(set(lbls_bn) - set(imgs_bn))
    print("missing label for image:", len(missing_lbl))
    print("missing image for label:", len(missing_img))

    class_count = load_class_count()
    if class_count is not None:
        print("class_count(from _class_map.json):", class_count)
    else:
        print("class_count: (no map found)")

    bad_format = []
    bad_parse = []
    out_of_range = []
    zero_or_neg = []
    huge_box = []
    bad_class = []
    empty_label = []
    obj_count_dist = Counter()
    class_dist = Counter()

    clipped_like = 0  # 0~1 바깥이었던 것 감지용(근사)
    total_objs = 0

    for bn, lp in lbls_bn.items():
        rows = parse_label_file(lp)
        if len(rows) == 0:
            empty_label.append(bn)
            obj_count_dist[0] += 1
            continue

        # 에러 줄 체크
        real_rows = []
        for r in rows:
            if isinstance(r[0], str):
                if r[0] == "BAD_FORMAT":
                    bad_format.append((bn, r[1]))
                else:
                    bad_parse.append((bn, r[1]))
            else:
                real_rows.append(r)

        obj_count_dist[len(real_rows)] += 1

        for cls, xc, yc, w, h in real_rows:
            total_objs += 1
            class_dist[cls] += 1

            # class range
            if class_count is not None and not (0 <= cls < class_count):
                bad_class.append((bn, cls))

            # range checks
            vals = [xc, yc, w, h]
            if any(v < 0.0 or v > 1.0 for v in vals):
                out_of_range.append((bn, cls, xc, yc, w, h))
            if any(v < 0.0 or v > 1.0 for v in vals):
                clipped_like += 1

            # non-positive w/h
            if w <= 0.0 or h <= 0.0:
                zero_or_neg.append((bn, cls, w, h))

            # huge boxes (경험상 0.9 넘으면 거의 이상치)
            if w > 0.9 or h > 0.9:
                huge_box.append((bn, cls, w, h))

    print("\n=== QUALITY SUMMARY ===")
    print("empty label files:", len(empty_label))
    print("bad_format lines:", len(bad_format))
    print("bad_parse lines:", len(bad_parse))
    print("out_of_range rows:", len(out_of_range))
    print("w/h <= 0 rows:", len(zero_or_neg))
    print("huge box rows(w>0.9 or h>0.9):", len(huge_box))
    print("bad class_id rows:", len(bad_class))
    print("total objects:", total_objs)

    print("\n=== OBJECT COUNT DIST (per image) ===")
    for k in sorted(obj_count_dist):
        print(f"{k} objects:", obj_count_dist[k])

    if len(out_of_range) > 0:
        print("\n[예시] out_of_range 5개:")
        for item in out_of_range[:5]:
            print(item)

    if len(empty_label) > 0:
        print("\n[예시] empty_label 5개:")
        for bn in empty_label[:5]:
            print(bn)

if __name__ == "__main__":
    main()
