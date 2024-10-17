import json
import pandas as pd
import re
from typing import List, Dict, Any

class PseudoLabeling:
    def __init__(self, path: str, inf_path: str, test_path: str, save_path: str, conf_thr: float = 0.3):
        self.path = path
        self.inf_path = inf_path # model output path
        self.test_path = test_path # test.json path
        self.save_path = save_path # pseudo labeledoutput path
        self.conf_thr = conf_thr # 신뢰도(Pseudo Labeling의 결과물이 믿을 만한가?)
        self.test_json = self._load_test_json()
        self.inf = self._load_inference_data()

    def _load_test_json(self) -> Dict[str, Any]:
        with open(self.test_path) as f:
            return json.load(f)


    # inference file's format : list([class, p(확률), p[0][0], p[0][1], p[1][0], p[1][1]])
    def _load_inference_data(self) -> List[List[str]]:
        return pd.read_csv(self.inf_path).values.tolist()

    @staticmethod
    def extract_number_from_filepath(filepath: str) -> int:
        '''
        test/0001.jpg -> 1
        test/0022.jpg -> 22
        '''
        filename = filepath.split('/')[-1].split('.')[0]
        numeric_part = re.search(r'\d+', filename).group()
        return int(numeric_part)

    def process_annotations(self) -> List[Dict[str, Any]]:
        annotations = []
        start_anno_id = 0

        for bbox, img_name in self.inf: # BBox, Id
            bboxes_splited = bbox.split()

            for i in range(6, len(bboxes_splited)+1, 6):
                bbox = bboxes_splited[i-6:i]
                # Class(Category), _, p[0][0], p[0][1], p[1][0], p[1][1]
                cla, conf, l, t, r, b = bbox[0], float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])
                width, height = r - l, b - t
                area = round(width * height, 2)

                if conf < self.conf_thr:
                    continue

                annotation = {
                    'image_id': self.extract_number_from_filepath(img_name),
                    'category_id': int(cla),
                    'area': area,
                    'bbox': [round(l, 1), round(t, 1), round(width, 1), round(height, 1)],
                    'iscrowd': 0,
                    'id': start_anno_id
                }
                annotations.append(annotation)
                start_anno_id += 1

        return annotations

    def run(self):
        annotations = self.process_annotations()
        self.test_json['annotations'] += annotations

        with open(self.save_path, 'w') as f:
            json.dump(self.test_json, f)

def main():
    path = '.' # Basic path
    inf_path = 'submission_faster_rcnn_epoch_30.csv'# Inference path
    test_path = f'{path}/test.json' # Pseudo Labeled test.json file
    save_path = f'{path}/psuedu_test.json' # Pseudo Labeled inference
    conf_thr = 0.3

    pseudo_labeler = PseudoLabeling(path, inf_path, test_path, save_path, conf_thr)
    pseudo_labeler.run()
    print(f"Pseudo labeling completed. Results saved to {save_path}")

if __name__ == "__main__":
    main()
