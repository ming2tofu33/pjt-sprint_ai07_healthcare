import os
import sys
import ast
import json
import shutil
import random
import numpy as np
import pandas as pd
import albumentations as A
import cv2
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# [경로 설정] ROOT/configs/data_config.py 로드
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from configs.data_config import CONFIG
except ImportError:
    print("🚨 configs/data_config.py를 찾을 수 없습니다!")
    raise

class PillPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.image_map = {}
        self.id_to_yolo = {}
        self.df = pd.DataFrame()

    # ---------------------------------------------------------
    # 0단계: [JSON 통합] 하위 폴더 수색 + 메타데이터 통합 + n개 객체 보존
    # ---------------------------------------------------------
    def step0_build_golden_csv(self):
        print("📊 [Step 0] 딥 폴더 수색 및 통합 장부 작성 시작...")
        os.makedirs(os.path.dirname(self.cfg["FINAL_CSV"]), exist_ok=True)
        all_rows = []
        g_ann_id, g_img_id = 1, 1

        sources = [("KAGGLE", self.cfg["KAGGLE_JSON"]), ("AIHUB", self.cfg["AIHUB_JSON"])]
        
        for src_name, json_root in sources:
            if not os.path.exists(json_root): continue
            
            json_list = []
            for root, _, files in os.walk(json_root):
                for f in files:
                    if f.lower().endswith('.json'):
                        json_list.append(os.path.join(root, f))

            for full_path in tqdm(json_list, desc=f"Scanning {src_name}"):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    first_img = data['images'][0]
                    true_cat_id = int(first_img['dl_idx'])
                    
                    # [표준화] 약제 명칭을 'pill_name'으로 통일하여 KeyError 방지
                    raw_pill_name = first_img.get('dl_name', first_img.get('dl_name_en', 'Unknown'))
                    
                    img_lookup = {img['id']: img for img in data['images']}
                    for ann in data.get('annotations', []):
                        if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                            img_info = img_lookup.get(ann['image_id'])
                            if not img_info: continue
                            
                            row_data = {
                                "annotation_id": g_ann_id, 
                                "image_id": g_img_id,
                                "category_id": true_cat_id, 
                                "pill_name": raw_pill_name,
                                "source": src_name,
                                "anno_bbox": str(ann['bbox'])
                            }
                            row_data.update(img_info)
                            all_rows.append(row_data)
                            g_ann_id += 1
                    g_img_id += 1
                except: continue
        
        if not all_rows: return False

        raw_df = pd.DataFrame(all_rows)
        # [중복 제거] 파일명과 좌표가 모두 같을 때만 중복으로 간주 (n개 객체 보존)
        self.df = raw_df.drop_duplicates(subset=['file_name', 'anno_bbox'], keep='first').copy()
        
        if 'id' in self.df.columns: self.df = self.df.drop(columns=['id'])
            
        self.df.to_csv(self.cfg["FINAL_CSV"], index=False, encoding='utf-8-sig')
        print(f"✅ 장부 생성 완료: {len(self.df)}건 저장.")
        return True

    # ---------------------------------------------------------
    # 1단계: [이상치 검수] 파손, 암흑, 하단 잘림 감지
    # ---------------------------------------------------------
    def _is_valid_image(self, path):
        img = cv2.imread(path)
        if img is None: return False, None
        if np.mean(img) < 5: return False, None
        
        h, w = img.shape[:2]
        if np.mean(img[int(h*0.9):, :, :]) < 3: return False, None
        return True, img

    # ---------------------------------------------------------
    # 5단계: [Metadata Mapping] 0점 방지용 통역 문서 생성
    # ---------------------------------------------------------
    def _prepare_mapping(self):
        print("📄 [Step 5] 통역 문서(class_mapping.csv) 생성 중...")
        os.makedirs(self.cfg["YOLO_ROOT"], exist_ok=True)
        mapping = self.df[['pill_name', 'category_id']].drop_duplicates('pill_name').sort_values('pill_name')
        mapping['yolo_id'] = range(len(mapping))
        mapping.to_csv(os.path.join(self.cfg["YOLO_ROOT"], "class_mapping.csv"), index=False)
        self.id_to_yolo = dict(zip(mapping['category_id'], mapping['yolo_id']))

    # ---------------------------------------------------------
    # 2단계: [YOLO 좌표 변환] 상대 비율 중심점 계산
    # ---------------------------------------------------------
    def step1_clean_and_yolo(self):
        print("\n🧼 [Step 1&2] 이상치 정제 및 YOLO 변환 시작...")
        self.image_map = {f: os.path.join(r, f) for r, _, fs in os.walk(self.cfg["SEARCH_ROOT"]) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        
        self.label_temp = os.path.join(self.cfg["YOLO_ROOT"], "temp_labels")
        os.makedirs(self.label_temp, exist_ok=True)
        
        valid_indices = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            img_path = self.image_map.get(row['file_name'])
            if not img_path: continue
            
            is_ok, img = self._is_valid_image(img_path)
            if not is_ok: continue
            
            try:
                bbox = ast.literal_eval(row['anno_bbox'])
                H, W = img.shape[:2]
                if bbox[0] < 0 or bbox[1] < 0 or (bbox[0]+bbox[2]) > W or (bbox[1]+bbox[3]) > H: continue
                
                # YOLO 정규화 공식: $$x_{c} = \frac{x + w/2}{W}, \quad y_{c} = \frac{y + h/2}{H}$$
                x_c, y_c = (bbox[0] + bbox[2]/2) / W, (bbox[1] + bbox[3]/2) / H
                wn, hn = bbox[2] / W, bbox[3] / H
                
                yolo_id = self.id_to_yolo[row['category_id']]
                txt_name = os.path.splitext(row['file_name'])[0] + ".txt"
                
                # 'a' 모드로 열어서 한 이미지 내 n개의 객체를 한 파일에 작성
                with open(os.path.join(self.label_temp, txt_name), 'a') as f:
                    f.write(f"{yolo_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}\n")
                valid_indices.append(idx)
            except: continue
        self.df = self.df.loc[valid_indices].copy()

    # ---------------------------------------------------------
    # 3단계: [Dataset Split] 8:2 스마트 분할 및 복사
    # ---------------------------------------------------------
    def step2_split_dataset(self):
        print("\n📦 [Step 3] Train/Val 8:2 분할 및 복사 중...")
        labels = [f for f in os.listdir(self.label_temp) if f.endswith('.txt')]
        if not labels: return
        
        train_fs, val_fs = train_test_split(labels, test_size=self.cfg["SPLIT_RATIO"], random_state=42)
        
        for fs, name in [(train_fs, "train"), (val_fs, "val")]:
            i_dest = os.path.join(self.cfg["YOLO_ROOT"], name, "images")
            l_dest = os.path.join(self.cfg["YOLO_ROOT"], name, "labels")
            os.makedirs(i_dest, exist_ok=True); os.makedirs(l_dest, exist_ok=True)
            
            for f in tqdm(fs, desc=name):
                shutil.move(os.path.join(self.label_temp, f), os.path.join(l_dest, f))
                base = os.path.splitext(f)[0]
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    if (base+ext) in self.image_map:
                        shutil.copy(self.image_map[base+ext], os.path.join(i_dest, base+ext))
                        break
        if os.path.exists(self.label_temp): shutil.rmtree(self.label_temp)

    # ---------------------------------------------------------
    # 4단계: [Targeted Augmentation] 부족한 데이터 보충 수업
    # ---------------------------------------------------------
    def step3_augment_train(self):
        print(f"\n🔥 [Step 4] 타겟 ID {self.cfg['AUG_TARGET_ID']} 증강 시작...")
        train_img_dir = os.path.join(self.cfg["YOLO_ROOT"], "train", "images")
        train_lbl_dir = os.path.join(self.cfg["YOLO_ROOT"], "train", "labels")
        if not os.path.exists(train_lbl_dir): return

        y_id = self.id_to_yolo.get(self.cfg["AUG_TARGET_ID"])
        if y_id is None: return

        # 증강 대상 파일 찾기
        target_files = []
        for f in os.listdir(train_lbl_dir):
            with open(os.path.join(train_lbl_dir, f), 'r') as lbl:
                if any(line.split()[0] == str(y_id) for line in lbl.readlines()):
                    target_files.append(f)

        needed = self.cfg["AUG_GOAL_COUNT"] - len(target_files)
        if needed <= 0:
            print("✨ 이미 목표 수량을 달성했습니다.")
            return

        # Albumentations 스위치 기반 설정
        augs = []
        if self.cfg["AUG_GEOMETRIC_ON"]:
            augs.extend([A.Rotate(limit=90, p=0.8), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
        if self.cfg["AUG_COLOR_ON"]:
            augs.extend([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5)])
        if self.cfg["AUG_BLUR_ON"]:
            augs.append(A.GaussianBlur(p=0.3))
            
        transform = A.Compose(augs, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        for i in tqdm(range(needed), desc="Augmenting"):
            src_lbl = random.choice(target_files)
            base = os.path.splitext(src_lbl)[0]
            
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(os.path.join(train_img_dir, base + ext)):
                    img_path = os.path.join(train_img_dir, base + ext); break
            if not img_path: continue
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with open(os.path.join(train_lbl_dir, src_lbl), 'r') as f:
                lines = f.readlines()
                bboxes = [list(map(float, l.split()[1:])) for l in lines]
                cls_labels = [int(l.split()[0]) for l in lines]

            augmented = transform(image=img, bboxes=bboxes, class_labels=cls_labels)
            
            new_name = f"aug_{i}_{base}"
            cv2.imwrite(os.path.join(train_img_dir, new_name + ".jpg"), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
            with open(os.path.join(train_lbl_dir, new_name + ".txt"), 'w') as f:
                for c, b in zip(augmented['class_labels'], augmented['bboxes']):
                    f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

    # ---------------------------------------------------------
    # 6단계: [YAML 생성] 학습용 지도 제작
    # ---------------------------------------------------------
    def generate_yaml(self):
        print("📄 [Step 6] 학습용 지도(data.yaml) 생성 중...")
        mapping_path = os.path.join(self.cfg["YOLO_ROOT"], "class_mapping.csv")
        if not os.path.exists(mapping_path): return

        mapping_df = pd.read_csv(mapping_path)
        class_names = mapping_df.sort_values('yolo_id')['pill_name'].tolist()
        
        yaml_data = {
            "path": self.cfg["YOLO_ROOT"],
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_names),
            "names": class_names
        }

        yaml_path = os.path.join(self.cfg["PROCESSED_DIR"], "data.yaml")
        os.makedirs(self.cfg["PROCESSED_DIR"], exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)
        print(f"✅ data.yaml 생성 완료: {yaml_path}")

    def run(self):
        print("🚀 [System] 올인원 파이프라인 가동!")
        if os.path.exists(self.cfg["YOLO_ROOT"]): shutil.rmtree(self.cfg["YOLO_ROOT"])
        
        if self.step0_build_golden_csv():
            self._prepare_mapping()
            self.step1_clean_and_yolo()
            self.step2_split_dataset()
            self.step3_augment_train()
            self.generate_yaml()
            print(f"\n✅ 모든 공정 완료! {self.cfg['YOLO_ROOT']}를 확인하세요.")

if __name__ == "__main__":
    pipeline = PillPipeline(CONFIG)
    pipeline.run()