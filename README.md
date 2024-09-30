
# 1. Introduction  
<br/>
<p align="center">
   <img src="./_img/main_ai_7.png" style="width:350px; height:70px;" />

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 7개월간의 교육과정입니다. 전체 과정은 level1~4로 구성되어 있으며 이 곳에는 그 중 첫 번째 대회인 `Image Classification`과제에 대한 **Level1 - 21조** 의 문제해결방법을 기록합니다.
  
<br/>

## 🎨 AI Palette  
”심장을 바쳐라”  
### 🔅 Members  

김한얼|김보현|김성주|윤남규|정수현|허민석
:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]
[Github](https://github.com/Haneol-Kijm)|[Github](https://github.com/boyamie)|[Github](https://github.com/kimmaru)|[Github](https://github.com/Tabianoo)|[Github](https://github.com/suhyun6363)|[Github](https://github.com/minseokheo)


### 🔅 Contribution  
`김한얼` &nbsp; Modeling • Feature Engineering • Age-specific model • EfficientNet Master • Out of fold  
`김보현` &nbsp; Dataset curation • Construct Pipeline • Mental Care • Data license verification  
`김성주` &nbsp; Dataset generation • Dataset curation • Mask synthesis • Hyperparameter tuning  
`윤남규` &nbsp; Team Management • Dataset preprocessing • Modeling • Make task-specific loss  
`정수현` &nbsp; EDA, Modeling • Visualizing • Search augmentation technique • MLops  
`허민석` &nbsp; Modeling • Active Learning • Mentoring • Huggingface pipeline • Handling imbalance problem  

[image1]: ./_img/김한얼.jpg
[image2]: ./_img/김보현.png
[image3]: ./_img/김성주.jpg
[image4]: ./_img/윤남규.png
[image5]: ./_img/정수현.png
[image6]: ./_img/허민석.jpg

<br/>

# 2. Project Outline  

![competition_title](./_img/competition_title.png)

<p align="center">
   <img src="./_img/mask_sample.png" width="300" height="300">
   <img src="./_img/class.png" width="300" height="300">
</p>

- Task : Image Classification
- Date : 2024.09.10 - 2024.09.26
- Description : 
- Image Resolution : 
- Train : 
- Test1 : 
- Test2 : 

### 🏆 Final Score  
<p align="center">
   <img src="./_img/final_score.png" width="700" height="90">
</p>

<br/>

# 3. Solution
![process][process]

### KEY POINT
-
-
-

&nbsp; &nbsp; → 주요 논점을 해결하는 방법론을 제시하고 실험결과를 공유하며 토론을 반복했습니다   

[process]: ./_img/process.png
<br/>

### Checklist
More Detail : https://github.com/jinmang2/boostcamp_ai_tech_2/blob/main/assets/ppt/palettai.pdf
- [x] Transformer based model
- [x] CNN based model(CLIP, EfficientNet, Nfnet, ResNet, ResNext)
- [ ] Age-specific model
- [ ] Three-head model
- [ ] External Dataset
- [x] Data Augmentation (Centorcrop, Resize)
- [ ] Focal loss
- [x] Weighted Sampling
- [x] Ensemble
- [ ] Out of fold
- [x] Test time augmentation
- [ ] Stacking
- [ ] Pseudo Labeling
- [ ] Noise Label Modification 
- [ ] Cutmix, cutout
- [ ] StyleGAN v2 + Mask Synthesis
- [x] Ray
- [ ] MC-Dropout
- [ ] Fixmatch
- [ ] Semi-supervised learning

### Evaluation

| Method | F-score |
| --- | --- |
| Synthetic Dataset + EfficientLite0 | 69.0 |
| Synthetic Dataset + non-prtrained BEIT | 76.9 |
| Synthetic Dataset + EfficientNet + Age-speicific | 76.9 |
| Synthetic Dataset + NFNet (Pseudo Labeling + Weighted Sampling)| 78.5 |
| Stacking BEIT + NFNet | 77.1 |

# 4. How to Use
- External dataset을 이용하기 위해서는 kaggle 의 https://www.kaggle.com/tapakah68/medical-masks-p4 에서 추가적으로 다운로드 받으셔야 합니다. 
```
.
├──input/data/train
├──input/data/eval
├──input/data/images(external kaggle data)
├──image-classification-level1-08
│   ├── configs
│   ├── solution
│         ├── cnn_engine
│         ├── hugging
│         ├── jisoo
│         ├── hugging
│         └── moon
```

- `soloution`안에는 각각 **train** •  **test** •  **inference**가 가능한 라이브러리가 들어있습니다  
- 사용자는 전체 코드를 내려받은 후, 옵션을 지정하여 개별 라이브러리의 모델을 활용할 수 있습니다
- 각 라이브러리의 구성요소는 `./solution/__main__.py`에서 확인할 수 있습니다  

### How to make Synthetic Dataset
- Use the repo Mask the face(https://github.com/aqeelanwar/MaskTheFace)
- Use the deepface to label age and gender(https://github.com/serengil/deepface)


```bash
git clone https://github.com/boostcampaitech2/image-classification-level1-08.git
```
```bash
$python __main__.py -m {module} -s {script} -c {config}

```
