import os
from datetime import datetime

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback


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
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        """
        테스팅 배치 종료 후 호출되는 함수

        Args:
            trainer: 트레이너 객체
            pl_module: PyTorch Lightning 모델
            outputs: 모델 출력
        """
        self.predictions.extend(outputs.cpu().numpy())

    def on_test_end(self, *args, **kwargs):
        """
        테스팅 종료 후 호출되는 함수
        """
        predictions = np.argmax(self.predictions.cpu().numpy(), axis=1)
        test_info = pd.read_csv(self.data_path)
        test_info["target"] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = os.path.join(
            self.ckpt_dir, f"{self.model_name}_predictions_{current_time}.csv"
        )
        test_info.to_csv(file_name, index=False, lineterminator="\n")
        print(f"Output csv file successfully saved in {file_name}!!")



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

    # def on_test_end(self, *args, **kwargs):
    #     """
    #     테스팅 종료 후 호출되는 함수
    #     """
    #     predictions = np.array(self.predictions)
    #     test_info = pd.read_csv(self.data_path)
    #     test_info["target"] = predictions
    #     test_info = test_info.reset_index().rename(columns={"index": "ID"})
    #     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    #     file_name = os.path.join(
    #         self.ckpt_dir, f"{self.model_name}_predictions_{current_time}.csv"
    #     )
    #     test_info.to_csv(file_name, index=False, lineterminator="\n")
    #     print(f"Output csv file successfully saved in {file_name}!!")
