import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import glob
import argparse
# pip install ensemble_boxes << 먼저 실행해주세요.

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--submission_path', required=True, help='submission file path') # submission file path
args = parser.parse_args()
# 앙상블 할 submission.csv 파일들이 담긴 폴더 경로 설정
submission_files = glob.glob(args.submission_path)
submission_df = [pd.read_csv(file) for file in submission_files]

image_ids = submission_df['image_id'].tolist()

# test.json 파일 경로 설정
annotation = '/data/ephemeral/home/dataset/test.json'
coco = COCO(annotation)

prediction_strings = []
file_names = []

for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]

    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()

        if len(predict_list)==0 or len(predict_list)==1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))

    if len(boxes_list):
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,iou_thr=0.55)

        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

    prediction_strings.append(prediction_string)
    file_names.append(image_id)


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names

# 최종 앙상블 submission 저장
submission.to_csv('/data/ephemeral/home/.submission/wbf_submission.csv', index=False)
