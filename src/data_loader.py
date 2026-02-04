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

# [경로 설정] ROOT/configs/data_config.py를 가져오기 위한 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from configs.data_config import CONFIG
except ImportError:
    print("configs/data_config.py를 찾을 수 없습니다!")
    raise

class PillPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.image_map = {}
        self.id_to_yolo = {}
        self.df = pd.DataFrame()

    # ---------------------------------------------------------
    # 0단계: [Deep JSON 통합] 하위 폴더 수색 및 결측/중복 제거
    # ---------------------------------------------------------
    def step0_build_golden_csv(self):
        print("[Step 0] 딥 폴더 수색 및 통합 장부 작성 시작...")
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
                    
                    img_lookup = {img['id']: img for img in data['images']}
                    for ann in data.get('annotations', []):
                        if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                            img_info = img_lookup.get(ann['image_id'])
                            if not img_info: continue
                            
                            row_data = {
                                "annotation_id": g_ann_id, "image_id": g_img_id,
                                "category_id": true_cat_id, "source": src_name,
                                "anno_bbox": str(ann['bbox'])
                            }
                            row_data.update(img_info)
                            all_rows.append(row_data)
                            g_ann_id += 1
                    g_img_id += 1
                except: continue
        
        if not all_rows: return False

        raw_df = pd.DataFrame(all_rows)
        self.df = raw_df.drop_duplicates(subset=['file_name'], keep='first').copy()
        
        if 'id' in self.df.columns: self.df = self.df.drop(columns=['id'])
            
        self.df.to_csv(self.cfg["FINAL_CSV"], index=False, encoding='utf-8-sig')
        print(f"장부 생성 완료: {len(self.df)}건 저장.")
        return True

    # ---------------------------------------------------------
    # 1단계: [이상치 검수] 이미지 파손, 암흑, 하단 잘림 감지
    # ---------------------------------------------------------
    def _is_valid_image(self, path):
        img = cv2.imread(path)
        if img is None: return False, None
        if np.mean(img) < 5: return False, None
        
        h, w = img.shape[:2]
        if np.mean(img[int(h*0.9):, :, :]) < 3: return False, None
        return True, img

    # ---------------------------------------------------------
    # 5단계: [Metadata Mapping] 통역 문서 생성
    # ---------------------------------------------------------
    def _prepare_mapping(self):
        print(" [Step 5] 통역 문서(class_mapping.csv) 생성 중...")
        os.makedirs(self.cfg["YOLO_ROOT"], exist_ok=True)
        mapping = self.df[['dl_name', 'category_id']].drop_duplicates('dl_name').sort_values('dl_name')
        mapping['yolo_id'] = range(len(mapping))
        mapping.to_csv(os.path.join(self.cfg["YOLO_ROOT"], "class_mapping.csv"), index=False)
        self.id_to_yolo = dict(zip(mapping['category_id'], mapping['yolo_id']))

    # ---------------------------------------------------------
    # 2단계: [YOLO 좌표 변환] 상대 비율 중심점 계산
    # ---------------------------------------------------------
    def step1_clean_and_yolo(self):
        print("\n [Step 1&2] 이상치 정제 및 YOLO 변환 시작...")
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
                
                # YOLO 정규화 공식 적용
                x_c, y_c = (bbox[0] + bbox[2]/2) / W, (bbox[1] + bbox[3]/2) / H
                wn, hn = bbox[2] / W, bbox[3] / H
                
                yolo_id = self.id_to_yolo[row['category_id']]
                txt_name = os.path.splitext(row['file_name'])[0] + ".txt"
                with open(os.path.join(self.label_temp, txt_name), 'a') as f:
                    f.write(f"{yolo_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}\n")
                valid_indices.append(idx)
            except: continue
        self.df = self.df.loc[valid_indices].copy()

    # ---------------------------------------------------------
    # 3단계: [Dataset Split] 8:2 스마트 분할
    # ---------------------------------------------------------
    def step2_split_dataset(self):
        print("\n [Step 3] Train/Val 8:2 분할 및 복사 중...")
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
        shutil.rmtree(self.label_temp)

    # ---------------------------------------------------------
    # 4단계: [Targeted Augmentation] 증강 로직
    # ---------------------------------------------------------
    def step3_augment_train(self):
        print(f"\n [Step 4] 타겟 ID {self.cfg['AUG_TARGET_ID']} 증강 시작...")
        # ... (생략된 증강 상세 로직 수행) ...
        pass

    # ---------------------------------------------------------
    # 마지막 단계: [YAML 생성] 학습용 지도 제작
    # ---------------------------------------------------------
    def generate_yaml(self):
        print(" [Step 6] 학습용 지도(data.yaml) 생성 중...")
        mapping_path = os.path.join(self.cfg["YOLO_ROOT"], "class_mapping.csv")
        if not os.path.exists(mapping_path):
            print("class_mapping.csv가 없습니다. YAML 생성을 중단합니다.")
            return

        mapping_df = pd.read_csv(mapping_path)
        class_names = mapping_df.sort_values('yolo_id')['pill_name'].tolist()
        
        yaml_data = {
            "path": self.cfg["YOLO_ROOT"],
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_names),
            "names": class_names
        }

        yaml_path = os.path.join(PROJECT_ROOT, "data.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)
        
        print(f"학습용 지도(data.yaml) 생성 완료: {yaml_path}")

    def run(self):
        print(" [System] 올인원 파이프라인 가동!")
        if os.path.exists(self.cfg["YOLO_ROOT"]): shutil.rmtree(self.cfg["YOLO_ROOT"])
        
        if self.step0_build_golden_csv():
            self._prepare_mapping()
            self.step1_clean_and_yolo()
            self.step2_split_dataset()
            self.step3_augment_train()
            self.generate_yaml() # 모든 공정 완료 후 YAML 생성
            print(f"\n 모든 공정 완료! {self.cfg['YOLO_ROOT']}를 확인하세요.")

if __name__ == "__main__":
    pipeline = PillPipeline(CONFIG)
    pipeline.run()