import os, json
from collections import defaultdict

ANN_ROOT = r"C:\DLP\sprint_ai_project1_data\train_annotations"
OUT_LABEL_ROOT = r"C:\DLP\sprint_ai_project1_data\train_labels_merged"

os.makedirs(OUT_LABEL_ROOT, exist_ok=True)

# 1) 이미지별로 (w,h)와 라벨들 모으기
img_wh = {}  # basename -> (w,h)
labels = defaultdict(list)  # basename -> list of (category_id, bbox)

category_names = {}  # category_id -> name (있으면 저장)

def read_json_utf8(path: str):
    # utf-8-sig까지 커버 (BOM 있는 파일도 안전)
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

# 2) 모든 json 순회
json_paths = []
for root, _, files in os.walk(ANN_ROOT):
    for fn in files:
        if fn.lower().endswith(".json"):
            json_paths.append(os.path.join(root, fn))

for jp in json_paths:
    data = read_json_utf8(jp)

    # 이 데이터셋은 images가 1개 들어있는 형태
    img = data["images"][0]
    file_name = img["file_name"]  # 예: ...png
    basename = os.path.splitext(file_name)[0]
    w, h = int(img["width"]), int(img["height"])
    img_wh[basename] = (w, h)

    # categories도 보통 1개(그 json의 알약 클래스)
    if "categories" in data and len(data["categories"]) > 0:
        cid = int(data["categories"][0]["id"])
        cname = data["categories"][0].get("name", str(cid))
        category_names[cid] = cname

    # annotations 안의 bbox들을 모은다
    for ann in data.get("annotations", []):
        cid = int(ann["category_id"])
        bbox = ann["bbox"]  # [x, y, bw, bh]
        labels[basename].append((cid, bbox))

# 3) category_id들을 YOLO class index(0..N-1)로 매핑
unique_cids = sorted(set(category_names.keys()) | {cid for b in labels for (cid, _) in labels[b]})
cid2idx = {cid: i for i, cid in enumerate(unique_cids)}

# 4) 이미지별로 YOLO txt 라벨 생성
missing_wh = []
for basename, ann_list in labels.items():
    if basename not in img_wh:
        missing_wh.append(basename)
        continue

    w, h = img_wh[basename]
    out_path = os.path.join(OUT_LABEL_ROOT, basename + ".txt")

    with open(out_path, "w", encoding="utf-8") as out:
        for cid, (x, y, bw, bh) in ann_list:
            cls = cid2idx[cid]

            # COCO -> YOLO (정규화)
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            ww = bw / w
            hh = bh / h

            # 안전 클리핑(가끔 라벨이 살짝 밖으로 나갈 수 있어서)
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)

            out.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

# 5) 매핑 저장(나중에 제출/해석할 때 필요)
map_path = os.path.join(OUT_LABEL_ROOT, "_class_map.json")
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "cid2idx": cid2idx,
            "idx2name": [category_names.get(cid, str(cid)) for cid in unique_cids],
            "idx2cid": unique_cids,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("완료!")
print("json 파일 수:", len(json_paths))
print("라벨 txt 생성 수:", len([fn for fn in os.listdir(OUT_LABEL_ROOT) if fn.endswith('.txt')]))
print	
print("클래스 개수:", len(unique_cids))
if missing_wh:
    print("WH 누락된 basename(이상):", missing_wh[:10])
