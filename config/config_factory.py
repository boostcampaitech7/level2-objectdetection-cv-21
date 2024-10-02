from .config import Config

# 모델 이름과 구성 설정 클래스를 매핑하는 디렉토리
CONFIG_MAP = {
    'ResNet': Config,
}


def get_config(model_name):
    """
    모델 이름에 해당하는 구성 설정 클래스를 반환합니다.

    Args:
        model_name (str): 모델 이름

    Returns:
        Config or subclass: 모델 이름에 해당하는 구성 설정 클래스

    Raises:
        KeyError: 모델 이름이 구성 설정 디렉토리에 없을 경우
    """
    # 모델 이름이 지정되지 않았을 경우 기본 모델인 ResNet18을 사용합니다.
    if model_name is None:
        model_name = "ResNet"

    try:
        # 모델 이름에 해당하는 구성 설정 클래스를 반환합니다.
        return CONFIG_MAP[model_name]
    except KeyError as exc:
        # 모델 이름이 구성 설정 디렉토리에 없을 경우 오류를 발생합니다.
        raise ValueError(f"Unsupported model name: {model_name}") from exc
