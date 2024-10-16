# Paper key point : Basic Pseudo Labeling
1. Object Detection Model들은 training을 통해서, 다수의 BBox를 생성합니다.
    -  그런데, 이 BBox들이 과연 "전부 믿을 수 있는가?"에 대한 의문이 논문의 핵심입니다.
    - 해당 논문에서는 Argmax(가장 믿을 수 있는 값) 들만을 True BBox로 정의합니다.
2. 그리고 나머지 BBox(not Argmax)들을 False BBox로 분류합니다.
3. (loop) 이렇게 분류된 True BBox를 이용해서, 다시 '1 -> 2' 과정을 다시 수행합니다.
4. 이렇게 반복적으로 training을 하면, 점점 더 True BBox를 얻을 수 있다는 것이 논문의 결론입니다.


# Paper key point : Cofidence Pseudo Labeling]
1. Basic Pseudo labeling은 argmax 연산을 통해서 Real True BBox를 찾아내는 것에 집중했습니다.
2. 그런데 여기서 질문이 생긴다. : "과연 이렇게 찾아낸 BBox를 정말 믿을 수 있는가? 즉, Argmax 연산을 통해서 찾아낸 BBox가 정말 맞는 것인가?"
3. 이에 대한 해답으로 Confidence(모델의 신뢰도)를 도입하는 것이 논문의 핵심입니다.
- 이를 위해서 True BBox 중에서 Confidence가 높은 경우만을 이용하는 과정이 추가됩니다.
- 이 방식은 모델이 옳다고 판단되는 경우에만 집중하므로, 보다 정확도를 높일 수 있다는 것이 논문의 결론입니다.
