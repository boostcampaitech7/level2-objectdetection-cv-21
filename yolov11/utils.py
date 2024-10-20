# utils.py
def check_and_adjust_dimensions(tensor):
    """
    주어진 텐서의 차원을 확인하여 필요한 경우 squeeze를 적용.
    :param tensor: 입력 텐서
    :return: 차원이 조정된 텐서
    """
    # 텐서의 차원이 1보다 크면 squeeze 적용
    if len(tensor.shape) > 1:
        tensor = tensor.squeeze(1)
    return tensor