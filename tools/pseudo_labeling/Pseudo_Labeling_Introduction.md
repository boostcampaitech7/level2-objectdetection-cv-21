# Background
- Object Detection을 수행한 inference file에서 row data의 예시는 다음과 같습니다.
```0 0.8031709 297.27094 420.27924 549.4864 793.03107```
- 해당 정보는 각각 "class", "p(가능성, 확률)", "BBox 꼭짓점들의 좌표"를 의미합니다.
- p 값이 높을수록, 모델은 해당 BBox의 존재 가능성이 높다고 판단합니다.

# Paper key point : Basic Pseudo Labeling
1. Object Detection Model들은 training을 통해서, 다수의 BBox를 생성합니다.
    -  그런데, "이 BBox들이 과연 전부 믿을 수 있는가?"에 대한 의문이 논문의 핵심입니다.
    - 해당 논문에서는 Argmax(가장 믿을 수 있는 값) 들만을 True BBox로 정의합니다.
2. 그리고 나머지 BBox(not Argmax)들을 False BBox로 분류합니다.
3. (loop) 이렇게 분류된 True BBox를 이용해서, '1 -> 2' 과정을 다시 수행합니다.
4. 이렇게 반복적으로 training을 하면, 점점 더 True BBox를 얻을 수 있다는 것이 논문의 결론입니다.


# Paper key point : Cofidence Pseudo Labeling (pseudo_labeling.py)
1. Basic Pseudo labeling은 argmax 연산을 통해서 Real True BBox를 찾아내는 것에 집중했습니다.
2. 그런데 여기서 질문이 생깁니다. : "과연 이렇게 찾아낸 True BBox를 정말 믿을 수 있는가? (계속되는 의심)"
3. 이에 대한 해답으로 Confidence(모델의 신뢰도)를 도입하는 것이 논문의 핵심입니다.
- 이를 위해서 True BBox 중에서 Confidence(신뢰도)가 threshold보다 높은 경우만을 이용하는 과정이 추가됩니다.
- 이 방식은 모델이 옳다고 판단되는 경우에만 집중하므로, 보다 정확도를 높일 수 있다는 것이 논문의 결론입니다.
