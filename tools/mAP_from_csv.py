import argparse
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_map_from_csv(csv_path, ann_path):
    """
    Compute mAP50 on the validation dataset with Pascal VOC format CSV file.

    Args:
    csv_path (str): Path to the CSV file in Pascal VOC format.
    ann_path (str): Path to the COCO annotation file (val2.json).
    class_num (int): Number of classes.

    Returns:
    mAP50 (float): Average precision at 50% IoU threshold.
    """
    # Load predictions from CSV file
    predictions = pd.read_csv(csv_path)

    # Load ground truth from COCO annotation file
    coco = COCO(ann_path)

    # Create detection results dictionary
    results = []
    for index, row in predictions.iterrows():
        image_id = row['image_id']
        prediction_string = row['PredictionString']

        # Parse prediction string
        predictions_list = prediction_string.split()
        num_predictions = len(predictions_list) // 6

        for i in range(num_predictions):
            cls_id = int(predictions_list[i * 6])
            confidence = float(predictions_list[i * 6 + 1])
            x_min = float(predictions_list[i * 6 + 2])
            y_min = float(predictions_list[i * 6 + 3])
            x_max = float(predictions_list[i * 6 + 4])
            y_max = float(predictions_list[i * 6 + 5])

            results.append({
                'image_id': image_id,
                'category_id': cls_id,
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'score': confidence
            })

    # Evaluate mAP50
    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [0.5]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble 기법을 수행합니다.')
    parser.add_argument('--path', default='./submssion.csv', help='submission file path')
    args = parser.parse_args()

    ann_path = '/data/ephemeral/home/dataset/val2.json'

    mAP50 = compute_map_from_csv(args.path, ann_path)
    print(f'mAP50: {mAP50:.4f}')
