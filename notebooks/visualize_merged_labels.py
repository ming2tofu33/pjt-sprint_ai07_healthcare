import os, glob, random
import matplotlib.pyplot as plt
from PIL import Image

IMG_DIR = r"C:\DLP\sprint_ai_project1_data\train_images"         # 실제 경로로!
LBL_DIR = r"C:\DLP\sprint_ai_project1_data\train_labels_merged"
OUT_DIR = r"C:\DLP\pjt-sprint_ai07_healthcare\debug_vis"
N = 20

IMG_EXTS = (".jpg", ".jpeg", ".png")
os.makedirs(OUT_DIR, exist_ok=True)

def list_images():
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(IMG_DIR, f"*{ext}"))
    return paths

def bn(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_labels(txt_path):
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cls, xc, yc, w, h = ln.split()
            rows.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return rows

def yolo_to_xyxy(xc, yc, w, h, W, H):
    # normalized -> pixel xyxy
    bw = w * W
    bh = h * H
    cx = xc * W
    cy = yc * H
    x1 = cx - bw/2
    y1 = cy - bh/2
    x2 = cx + bw/2
    y2 = cy + bh/2
    return x1, y1, x2, y2

def main():
    imgs = list_images()
    random.shuffle(imgs)
    imgs = imgs[:N]

    for ip in imgs:
        name = bn(ip)
        tp = os.path.join(LBL_DIR, name + ".txt")
        if not os.path.exists(tp):
            continue

        im = Image.open(ip).convert("RGB")
        W, H = im.size
        labels = load_labels(tp)

        plt.figure()
        plt.imshow(im)
        ax = plt.gca()

        for cls, xc, yc, w, h in labels:
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, str(cls), fontsize=10)

        plt.axis("off")
        out_path = os.path.join(OUT_DIR, name + "_vis.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print("saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
