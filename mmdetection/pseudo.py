from mmdet.apis import DetInferencer
import json
import pandas as pd
import re
# pth 파일로 test.json의 이미지들 하나씩 inference
# test.json의 annotation에 image_id, category_id, area, bbox를 저장

# /data/ephemeral/home/mmdetection/mmdet/apis/det_inferencer.py에서 pred_score_thr 변경

path = './'
inf_path = './submission_faster_rcnn_epoch_30.csv' # inference한 json

test_path = f'{path}/test.json' # test json
save_path = f'{path}/psuedu_test.json' # 저장


conf_thr = 0.3

with open(test_path) as f:
    test_json = json.load(f)

inf = pd.read_csv(inf_path).values.tolist()

annotations = []
start_anno_id = 0

def extract_number_from_filepath(filepath):
    '''
    test/0001.jpg -> 1
    test/0022.jpg -> 22
    '''
    filename = filepath.split('/')[-1].split('.')[0] # 1
    numeric_part = re.search(r'\d+', filename).group() # file name list
    return int(numeric_part)

for bbox, img_name in inf:
    bboxes_splited = bbox.split()
    num_bbox = 0


    for i in range(6, len(bboxes_splited)+1, 6):
        bbox = bboxes_splited[i-6:i]
        cla, conf, l, t, r, b = bbox[0], float(bbox[1]), float(bbox[2]), \
                                                float(bbox[3]), float(bbox[4]), float(bbox[5])
        width, height = r - l, b - t
        area = round(width * height, 2)


        if conf < conf_thr:
            continue

        annotation = dict()
        annotation['image_id'] = extract_number_from_filepath(img_name)
        annotation['category_id'] = int(cla)
        annotation['area'] = area
        annotation['bbox'] = [round(l, 1), round(t, 1), round(width, 1), round(height, 1)]
        annotation['iscrowd'] = 0
        annotation['id'] = start_anno_id
        annotations.append(annotation)
        start_anno_id += 1


test_json['annotations'] += annotations

with open(save_path, 'w') as f:
    json.dump(test_json, f)
