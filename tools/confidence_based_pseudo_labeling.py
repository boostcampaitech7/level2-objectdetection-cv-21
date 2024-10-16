from mmdet.apis import DetInferencer
import json
import pandas as pd
import re
# pth 파일로 test.json의 이미지들 하나씩 inference
# test.json의 annotation에 image_id, category_id, area, bbox를 저장

# /data/ephemeral/home/mmdetection/mmdet/apis/det_inferencer.py에서 pred_score_thr 변경
class pseudo_labeling() :
path = '/data/ephemeral/home/dataset'
inf_path = '/data/ephemeral/home/dino234_cascade_all ensemble.csv' # "(!) Inference한 json"
test_path = f'{path}/test.json' # test json
save_path = f'{path}/psuedu_test.json' # 저장

conf_thr = 0.3 # 신뢰도 (추가 조사 필요)

with open(test_path) as f:
    test_json = json.load(f)

''' [Ln 24]
- Inference된 image들의 list
- "[BBox, img_name(Id)]" 로 구성됨.
'''
inf = pd.read_csv(inf_path).values.tolist()

annotations = []
start_anno_id = 0

def extract_number_from_filepath(filepath):
    '''
    test/0001.jpg -> 1
    test/0022.jpg -> 22
    '''
    filename = filepath.split('/')[-1].split('.')[0]
    numeric_part = re.search(r'\d+', filename).group()
    return int(numeric_part)


for bbox, img_name in inf: # BBox, Id
    bboxes_splited = bbox.split() # BBox를 하나씩 추출
    num_bbox = 0


    for i in range(6, len(bboxes_splited)+1, 6): #
        bbox = bboxes_splited[i-6:i]
        '''
        Class(Category), _, p[0][0], p[0][1], p[1][0], p[1][1]
        '''
        cla, conf, l, t, r, b = bbox[0], float(bbox[1]), float(bbox[2]), \
                                                float(bbox[3]), float(bbox[4]), float(bbox[5])
        width, height = r - l, b - t # w, h
        area = round(width * height, 2)


        if conf < conf_thr: # conf < 신뢰도
            continue

        annotation = dict()
        annotation['image_id'] = extract_number_from_filepath(img_name)
        annotation['category_id'] = int(cla)
        annotation['area'] = area
        annotation['bbox'] = [round(l, 1), round(t, 1), round(width, 1), round(height, 1)]
        annotation['iscrowd'] = 0 # default
        annotation['id'] = start_anno_id
        annotations.append(annotation)
        start_anno_id += 1


test_json['annotations'] += annotations

with open(save_path, 'w') as f:
    json.dump(test_json, f)
