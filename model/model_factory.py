# 모델을 생성하기 위한 팩토리 클래스
import importlib


# 모델 생성 함수
# 모델명과 파라미터를 받아서 모델 객체를 반환한다.
def create_model(model_name="ResNet18", **kwargs):
    # 동적으로 모델 모듈을 로드한다.
    model_module = importlib.import_module(f"model.{model_name}")

    # 해당 모델 클래스를 가져온다.
    model_class = getattr(model_module, model_name)

    # 모델 객체를 생성하여 반환한다.
    model = model_class(**kwargs)
    return model
