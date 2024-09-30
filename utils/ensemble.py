import os
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from model import LightningModule
from lightning import Trainer

from dataset import get_test_loader
from engine import PredictionEnsembleCallback
class EnsemblePredictor:
    def __init__(self, config):
        """
        Args:
            config: 모델 및 실험 설정이 포함된 configuration 객체
            ckpt_dir (str): 체크포인트 디렉토리
            device (str): 연산에 사용할 디바이스
        """
        self.config = config
        self.method = config.experiment.ensemble_method
        self.ckpt_dir = config.experiment.ensemble_path
        self.test_loader = get_test_loader(config)

    def load_models(self):
        # 체크포인트에서 모델 로드
        models = []
        for ckpt_file in os.listdir(self.ckpt_dir):
            if ckpt_file.endswith(".ckpt"):
                checkpoint = torch.load(ckpt_file)
                hparam = checkpoint.hparam
                self.config.model.model_name = hparam.model_name
                model = LightningModule.load_from_checkpoint(os.path.join(self.ckpt_dir, ckpt_file), config=self.config.model)
                models.append(model)
        return models

    # 앙상블 예측
    def ensemble_predict(self, models, dataloader):
        # 예측값을 저장할 배열을 초기화 합니다.
        ensemble_predictions = []
        for i, model in enumerate(models):
            trainer = Trainer(devices=1, precision='16-mixed')
            callback = PredictionEnsembleCallback()
            trainer.callbacks.append(callback)
            trainer.test(model, dataloaders=dataloader)
            # print(len(callback.predictions)) [[500, 10014]]
            prediction_tensor = torch.stack(callback.predictions)
            print(f"{i}th model prediction output shape is {prediction_tensor.shape}")
            ensemble_predictions.append(prediction_tensor)
        # 예측값을 합산하여 앙상블 합니다.
        ensemble_output = torch.stack(ensemble_predictions).mean(dim=0)
        print(f"Ensembled shape is {ensemble_output.shape}")
        # 최종 예측값을 반환합니다.
        return ensemble_output

    def ensemble(self):
        models = self.load_models()
        if self.method == 'uniform_soup':
            model = self.uniform_soup(models)
        elif self.method == 'greedy_soup':
            model = self.greedy_soup(models)
        elif self.method == 'ensemble_predict':
            pass
        else:
            raise ValueError("Invalid ensemble method")



        if self.method == 'ensemble_predict':
            predictions = self.ensemble_predict(models, self.test_loader)
        else:
            predictions = model.predict(self.test_loader)
        
        self.save_to_csv(predictions=predictions)

    def save_to_csv(self, predictions):
        # 예측 결과를 csv 파일로 저장
        test_info = pd.read_csv(self.config.dataset.data_path + "/test.csv")
        test_info["target"] = np.argmax(predictions.cpu().numpy(), axis=1)
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        file_name = os.path.join(
            self.ckpt_dir, f"ensemble_predictions_{current_time}.csv"
        )
        test_info.to_csv(file_name, index=False, lineterminator="\n")
        print(f"Output csv file successfully saved in {file_name}!!")

    # Run the ensemble predictor using the specified method
    def run(self):
        self.ensemble()

