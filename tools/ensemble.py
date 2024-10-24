import pandas as pd
import argparse
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion

def parse_prediction_string(pred_string):
    """예측 문자열을 파싱하여 boxes, scores, labels 리스트로 변환"""
    if pd.isna(pred_string):
        return [], [], []

    predictions = pred_string.strip().split()
    num_predictions = len(predictions) // 6

    boxes = []
    scores = []
    labels = []

    for i in range(num_predictions):
        start_idx = i * 6
        label = int(float(predictions[start_idx]))
        score = float(predictions[start_idx + 1])
        # bbox 좌표 [x1, y1, x2, y2] 추출 및 정규화
        bbox = [float(x) for x in predictions[start_idx + 2:start_idx + 6]]

        # 이미지 크기로 정규화 (0~1 범위로 변환)
        # 여기서는 예시로 1024x1024 이미지 크기를 가정
        normalized_bbox = [
            bbox[0] / 1024,  # x1
            bbox[1] / 1024,  # y1
            bbox[2] / 1024,  # x2
            bbox[3] / 1024   # y2
        ]

        boxes.append(normalized_bbox)
        scores.append(score)
        labels.append(label)

    return boxes, scores, labels

def format_prediction_string(boxes, scores, labels):
    """박스, 점수, 라벨을 예측 문자열 형식으로 변환"""
    predictions = []
    for label, score, box in zip(labels, scores, boxes):
        # 박스 좌표를 다시 원래 스케일로 변환
        denorm_box = [
            box[0] * 1024,  # x1
            box[1] * 1024,  # y1
            box[2] * 1024,  # x2
            box[3] * 1024   # y2
        ]
        pred = f"{int(label)} {score:.8f} {' '.join([f'{x:.5f}' for x in denorm_box])}"
        predictions.append(pred)

    return ' '.join(predictions)

def apply_ensemble(csv_files, nms_thr=0.1, wbf_thr=0.3, skip_box_thr=0.0001, nms=False, soft_nms=False, wbf=False):
    """앙상블 적용"""
    # 각 모델별 예측 데이터프레임을 하나로 병합
    pred_dfs = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(pred_dfs, ignore_index=True)

    # 결과를 저장할 새로운 DataFrame 생성
    result_df = pd.DataFrame(columns=['PredictionString', 'image_id'])

    # 각 이미지별로 처리
    for image_id in df['image_id'].unique():
        image_preds = df[df['image_id'] == image_id]

        # 각 예측에서 박스, 점수, 라벨 추출
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred_string in image_preds['PredictionString']:
            boxes, scores, labels = parse_prediction_string(pred_string)
            if boxes:  # 빈 예측이 아닌 경우에만 추가
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)

        if not boxes_list:  # 예측이 없는 경우
            result_df = pd.concat([result_df, pd.DataFrame({
                'PredictionString': [''],
                'image_id': [image_id]
            })], ignore_index=True)
            continue

        # NMS 적용
        if nms:
            nms_boxes_list = []
            nms_scores_list = []
            nms_labels_list = []
            # Looping over boxes list of each models
            for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
                boxes, scores, labels = nms(
                [boxes],
                [scores],
                [labels],
                weights=None,
                iou_thr=nms_thr,
                skip_box_thr=skip_box_thr
                )
                nms_boxes_list.append(boxes)
                nms_scores_list.append(scores)
                nms_labels_list.append(labels)
            boxes_list = nms_boxes_list
            scores_list = nms_scores_list
            labels_list = nms_labels_list

        # 또는 Soft-NMS 적용
        elif soft_nms:
            nms_boxes_list = []
            nms_scores_list = []
            nms_labels_list = []
            # Looping over boxes list of each models
            for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
                boxes, scores, labels = soft_nms(
                [boxes],
                [scores],
                [labels],
                weights=None,
                iou_thr=nms_thr,
                skip_box_thr=skip_box_thr
                )
                nms_boxes_list.append(boxes)
                nms_scores_list.append(scores)
                nms_labels_list.append(labels)
            boxes_list = nms_boxes_list
            scores_list = nms_scores_list
            labels_list = nms_labels_list

        # WBF 적용
        if wbf:
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=None,
                iou_thr=wbf_thr,
                skip_box_thr=skip_box_thr
            )
        else:
            boxes = [box for boxes in boxes_list for box in boxes]
            scores = [score for scores in scores_list for score in scores]
            labels = [label for labels in labels_list for label in labels]


        # 결과를 예측 문자열 형식으로 변환
        prediction_string = format_prediction_string(boxes, scores, labels)

        # 결과 DataFrame에 추가
        result_df = pd.concat([result_df, pd.DataFrame({
            'PredictionString': [prediction_string],
            'image_id': [image_id]
        })], ignore_index=True)

    return result_df

if __name__ == "__main__":
    # WBF 앙상블 수행
    parser = argparse.ArgumentParser(description='Ensemble 기법을 수행합니다.')
    parser.add_argument('--csv-files', nargs='+', help='submission file path')
    parser.add_argument('--nms',action='store_true', help='whether to apply nms')
    parser.add_argument('--soft-nms',action='store_true', help='whether to apply soft-nms')
    parser.add_argument('--wbf',action='store_true', help='whether to apply wbf')
    parser.add_argument('--nms_thr', type=float, default=0.5, help='nms threshold which is used in nms and soft-nms')
    parser.add_argument('--wbf_thr', type=float, default=0.6, help='wbf iou threshold')
    args = parser.parse_args()
    if args.soft_nms and args.nms:
        raise ValueError('soft-nms and nms cannot be applied at the same time')
    result_df = apply_ensemble(
        args.csv_files,
        nms_thr=args.nms_thr,
        wbf_thr=args.wbf_thr,          # IoU 임계값. 겹치는 BBox가 많을수록 낮은 값을 선택해야 합니다.
        skip_box_thr=0.0001,   # 최소 confidence score
        nms=args.nms,
        soft_nms=args.soft_nms,
        wbf=args.wbf
    )

    # 결과 저장
    file_name=f"submission_nms_{args.nms_thr}_wbf_{args.wbf_thr}.csv"
    result_df.to_csv(file_name, index=False)
    print(f"WBF 앙상블이 완료되었습니다. 결과가 f{file_name}에 저장되었습니다.")
