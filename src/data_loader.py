import os
import sys
import shutil
import ast
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 경로 설정
current_file_path = os.path.abspath(__file__)
preprocessing_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(preprocessing_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path: sys.path.append(project_root)

# Config 로드
try:
    from configs.data_config import CONFIG
except ImportError:
    print("🚨 [Error] configs/data_config.py 없음")
    sys.exit(1)

class IntegratedDataLoader:
    def __init__(self):
        self.cfg = CONFIG
        
        # 1. CSV 로드
        if os.path.exists(self.cfg["RAW_CSV"]):
            print(f"📄 CSV 로드: {os.path.basename(self.cfg['RAW_CSV'])}")
            self.df = pd.read_csv(self.cfg["RAW_CSV"])
        else:
            raise FileNotFoundError(f"CSV 없음: {self.cfg['RAW_CSV']}")

        # 2. 이미지 탐색
        self.image_map = self._find_all_images(self.cfg["SEARCH_ROOT"])
        
        # 3. 폴더 준비
        self.label_dir = os.path.join(self.cfg["OUTPUT_ROOT"], "temp_labels")
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.cfg["OUTPUT_ROOT"], exist_ok=True)
        
        # 4. 매핑 생성
        self._create_mapping()

    def _find_all_images(self, search_root):
        print(f"🔍 [Finder] 이미지 수색 중... ({search_root})")
        image_map = {} 
        count = 0
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
                    image_map[file] = os.path.join(root, file)
                    count += 1
        print(f"   👉 총 {count}장 발견!")
        return image_map

    def _create_mapping(self):
        name_col = 'dl_name_en' if 'dl_name_en' in self.df.columns else 'img_dl_name'
        id_col = 'category_id' if 'category_id' in self.df.columns else 'img_dl_idx'
        try:
            mapping_info = self.df[[name_col, id_col]].drop_duplicates(name_col).sort_values(name_col)
            self.mapping_df = pd.DataFrame({
                'pill_name': mapping_info[name_col].values,
                'true_dl_idx': mapping_info[id_col].values
            })
            self.mapping_df['yolo_id'] = range(len(self.mapping_df))
            self.id_to_yolo = dict(zip(self.mapping_df['true_dl_idx'], self.mapping_df['yolo_id']))
            self.mapping_df.to_csv(os.path.join(self.cfg["OUTPUT_ROOT"], "class_mapping.csv"), index=False)
        except KeyError:
            print(f"🚨 컬럼 에러: {self.df.columns.tolist()}")
            raise

    # 🚀 Step 1: YOLO 변환
    def step1_clean_and_yolo(self):
        print("\n🧼 [Step 1] YOLO 변환...")
        bbox_col = 'bbox' if 'bbox' in self.df.columns else 'anno_bbox'
        fname_col = 'file_name' if 'file_name' in self.df.columns else 'img_file_name'
        cat_col = 'category_id' if 'category_id' in self.df.columns else 'img_dl_idx'
        
        valid_count = 0
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            fname = row[fname_col]
            img_path = self.image_map.get(fname)
            if img_path is None: continue
            try:
                bbox = row[bbox_col]
                if isinstance(bbox, str): bbox = ast.literal_eval(bbox)
                bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                
                if 'width' in row and 'height' in row and row['width'] > 0:
                    W, H = row['width'], row['height']
                else:
                    img = cv2.imread(img_path)
                    if img is None: continue
                    H, W, _ = img.shape
                
                # 안전 로직
                if bw > W or bh > H: continue 
                if bx < 0 or by < 0: continue
                if bx + bw > W or by + bh > H: continue

                cat_id = row[cat_col]
                yolo_class_id = self.id_to_yolo.get(cat_id, 0)
                self._save_yolo_label(fname, yolo_class_id, [bx, by, bw, bh], W, H)
                valid_count += 1
            except: continue
        print(f"✅ 변환 완료: {valid_count}장")

    def _save_yolo_label(self, filename, class_id, bbox, W, H):
        bx, by, bw, bh = bbox
        x_center, y_center = (bx + bw/2) / W, (by + bh/2) / H
        w_norm, h_norm = bw / W, bh / H
        x_center, y_center = min(max(x_center, 0.0), 1.0), min(max(y_center, 0.0), 1.0)
        w_norm, h_norm = min(max(w_norm, 0.0), 1.0), min(max(h_norm, 0.0), 1.0)
        txt_name = os.path.splitext(os.path.basename(filename))[0] + ".txt"
        with open(os.path.join(self.label_dir, txt_name), 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # 🚀 Step 2: 분할
    def step2_split_dataset(self):
        print("\n📦 [Step 2] Train/Val 분할...")
        label_files = [f for f in os.listdir(self.label_dir) if f.endswith('.txt')]
        if not label_files: return
        train_fs, val_fs = train_test_split(label_files, test_size=self.cfg["SPLIT_RATIO"], random_state=42)
        self._move_files(train_fs, "train")
        self._move_files(val_fs, "val")
        if os.path.exists(self.label_dir): shutil.rmtree(self.label_dir)

    def _move_files(self, files, split):
        dest_img = os.path.join(self.cfg["OUTPUT_ROOT"], split, "images")
        dest_lbl = os.path.join(self.cfg["OUTPUT_ROOT"], split, "labels")
        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_lbl, exist_ok=True)
        for lbl_f in tqdm(files, desc=f"To {split}"):
            src_lbl = os.path.join(self.label_dir, lbl_f)
            shutil.move(src_lbl, os.path.join(dest_lbl, lbl_f))
            base = os.path.splitext(lbl_f)[0]
            img_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                full_name = base + ext
                if full_name in self.image_map:
                    shutil.copy(self.image_map[full_name], os.path.join(dest_img, full_name))
                    img_found = True
                    break
            if not img_found:
                os.remove(os.path.join(dest_lbl, lbl_f))

    # 🚀 Step 3: 증강 (풀옵션 리모컨 적용)
    def step3_augment_train(self):
        if not self.cfg.get("USE_AUGMENTATION", False):
            print("\n🛑 [Step 3] 증강 OFF")
            return

        target_id = self.cfg['AUG_TARGET_ID']
        print(f"\n🔥 [Step 3] 증강 시작 (Target ID: {target_id})")
        
        train_lbl_dir = os.path.join(self.cfg["OUTPUT_ROOT"], "train", "labels")
        train_img_dir = os.path.join(self.cfg["OUTPUT_ROOT"], "train", "images")
        if not os.path.exists(train_lbl_dir): return
        
        target_files = []
        for lbl_f in os.listdir(train_lbl_dir):
            try:
                with open(os.path.join(train_lbl_dir, lbl_f), 'r') as f:
                    if any(line.split()[0] == str(target_id) for line in f.readlines()):
                        target_files.append(lbl_f)
            except: continue
        
        needed = self.cfg["AUG_COUNT"] - len(target_files)
        if needed <= 0:
            print("   👉 증강 불필요")
            return

        # =================================================================
        # 🎮 [풀옵션 증강 파이프라인] Config 값 적용 (기본값: false)
        #      -> false 적용시 cfg.get("AUG_...", 10) -> 숫자로 적용 됨
        # =================================================================
        print(f"   🎛️ 증강 파라미터 적용 중...")
        
        transform = A.Compose([
            # 1. 기하학 (Rotate, Flip) - 리모컨 적용
            A.Rotate(limit=self.cfg.get("AUG_ROTATE_LIMIT", 30), 
                     p=self.cfg.get("AUG_ROTATE_PROB", 0.7)),
                     
            A.HorizontalFlip(p=self.cfg.get("AUG_FLIP_PROB", 0.5)),
            
            # 2. 밝기 (Brightness) - 리모컨 적용
            A.RandomBrightnessContrast(
                brightness_limit=self.cfg.get("AUG_BRIGHT_LIMIT", 0.2), 
                contrast_limit=self.cfg.get("AUG_BRIGHT_LIMIT", 0.2), 
                p=self.cfg.get("AUG_BRIGHT_PROB", 0.5)
            ),
            
            # 3. 색조 (Hue/Sat/Val) - 리모컨 적용
            A.HueSaturationValue(
                hue_shift_limit=self.cfg.get("AUG_HUE_LIMIT", 20), 
                sat_shift_limit=self.cfg.get("AUG_SAT_LIMIT", 30), 
                val_shift_limit=self.cfg.get("AUG_VAL_LIMIT", 20), 
                p=self.cfg.get("AUG_HSV_PROB", 0.5)
            ),
            
            # 4. RGB Shift - 리모컨 적용
            A.RGBShift(
                r_shift_limit=self.cfg.get("AUG_RGB_SHIFT", 15), 
                g_shift_limit=self.cfg.get("AUG_RGB_SHIFT", 15), 
                b_shift_limit=self.cfg.get("AUG_RGB_SHIFT", 15), 
                p=self.cfg.get("AUG_RGB_PROB", 0.5)
            ),
            
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))
        # =================================================================

        for i in tqdm(range(needed), desc="Augmenting"):
            try:
                src_lbl_f = random.choice(target_files)
                base = os.path.splitext(src_lbl_f)[0]
                src_img_f = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    if os.path.exists(os.path.join(train_img_dir, base + ext)):
                        src_img_f = base + ext
                        break
                if not src_img_f: continue
                img = cv2.imread(os.path.join(train_img_dir, src_img_f))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with open(os.path.join(train_lbl_dir, src_lbl_f), 'r') as f:
                    lines = f.readlines()
                bboxes, cls_labels = [], []
                for line in lines:
                    parts = list(map(float, line.split()))
                    cls_labels.append(int(parts[0]))
                    bboxes.append(parts[1:]) 
                augmented = transform(image=img, bboxes=bboxes, class_labels=cls_labels)
                new_base = f"aug_{i}_{base}"
                cv2.imwrite(os.path.join(train_img_dir, new_base + ".jpg"), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                with open(os.path.join(train_lbl_dir, new_base + ".txt"), 'w') as f:
                    for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            except: continue

    def run(self):
        print("🚀 [Integrated Data Loader] 가동!")
        if os.path.exists(self.cfg["OUTPUT_ROOT"]):
            try: shutil.rmtree(self.cfg["OUTPUT_ROOT"])
            except: pass
        self.step1_clean_and_yolo()
        self.step2_split_dataset()
        self.step3_augment_train()
        print(f"\n✨ 완료! 결과물: {self.cfg['OUTPUT_ROOT']}")
        print(f"   📊 증강 사용: {self.cfg.get('USE_AUGMENTATION', False)}")

if __name__ == "__main__":
    loader = IntegratedDataLoader()
    loader.run()