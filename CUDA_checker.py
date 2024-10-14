# 서버의 개발 환경을 위해서, 특정 library가 CUDA 버전인지 확인하는 파일 입니다.
# 해당 예제는 "cv2"를 확인했습니다. 확인할 library를 cv2 대신에 이용해주세요.
import cv2
import numpy as np

# library 버전 출력
print(f"OpenCV 버전: {cv2.__version__}")

# libary의 CUDA 지원 여부를 확인합니다.
print(f"CUDA 사용 가능: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

# CUDA를 지원하는 경우에만, GPU 정보를 출력합니다.
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("\nGPU 정보:")
    print(cv2.cuda.printCudaDeviceInfo(0))

# CUDA 지원 여부에 따른 메시지 출력
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("\n CUDA를 사용중입니다.")
else:
    print("\n CUDA를 사용하고 있지 않습니다.")

# libary의 빌드 정보 출력
print("\nOpenCV 빌드 정보:")
print(cv2.getBuildInformation())
