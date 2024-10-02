import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
from pycocotools.coco import COCO


class PredictionCallback(Callback):
    """
    모델 테스팅에 필요한 콜백 함수

    Args:
        data_path (str): 테스팅 데이터 경로
        ckpt_dir (str): 체크포인트 디렉토리
        model_name (str): 모델 이름
    """

    def __init__(self, data_path, ckpt_dir, model_name):
        """
        콜백 함수 초기화
        """
        self.data_path = data_path
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.outputs = []

    def on_test_batch_end(self, trainer, outputs, *args, **kwargs):
        """
        테스팅 배치 종료 후 호출되는 함수

        Args:
            trainer: 트레이너 객체
            pl_module: PyTorch Lightning 모델
            outputs: 모델 출력
        """
        self.outputs.extend(outputs)

    def on_test_end(self, *args, **kwargs):
        """
        테스팅 종료 후 호출되는 함수
        """
        score_threshold = 0.05
        prediction_strings = []
        file_names = []
        annotation_path = os.path.join(self.data_path, "test.json")
        coco = COCO(annotation_path)

        # submission 파일 생성
        for i, output in enumerate(self.outputs):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > score_threshold: 
                    # label[1~10] -> label[0~9]
                    prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                        box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_path = os.path.join(
            self.ckpt_dir, f"{self.model_name}_predictions_{current_time}.csv"
        )
        submission = pd.DataFrame()
        submission['PredictionString'] = self.prediction_strings
        submission['image_id'] = self.file_names
        submission.to_csv(output_path, index=None, lineterminator="\n")
        print(submission.head())
        print(f"Output csv file successfully saved in {output_path}!!")


class PredictionEnsembleCallback(Callback):
    def __init__(self):
        """
        콜백 함수 초기화
        """
        # self.data_path = data_path
        # self.ckpt_dir = ckpt_dir
        # self.model_name = model_name
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        """
        테스팅 배치 종료 후 호출되는 함수

        Args:
            trainer: 트레이너 객체
            pl_module: PyTorch Lightning 모델
            outputs: 모델 출력
        """
        self.predictions.extend(outputs)