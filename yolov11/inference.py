def convert_yolo(coco_json, image_dir):
    """
    COCO 형식의 JSON 파일을 YOLO 형식의 .txt 라벨 파일로 변환하는 함수
    :param coco_json: COCO JSON 파일 경로
    :param image_dir: 이미지와 .txt 파일이 저장될 동일한 디렉토리
    """
    # COCO JSON 파일 로드
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # test 디렉토리에 있는 jpg 파일 목록 가져오기
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')}

    # 각 이미지에 대해 YOLO 형식의 텍스트 파일 생성
    for img in coco_data['images']:
        img_id = img['id']
        img_file = os.path.splitext(img['file_name'])[0]  # 확장자 제거한 파일명
        width = img['width']
        height = img['height']

        # 해당 이미지가 test 디렉토리에 있을 때만 변환
        if img_file not in image_files:
            continue

        # .txt 파일 생성 경로 설정
        label_file = img_file + '.txt'
        label_path = os.path.join(image_dir, label_file)

        # .txt 파일을 저장할 디렉토리가 있는지 확인
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # 해당 이미지에 대한 라벨 파일 생성
        with open(label_path, 'w') as label_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id']  # YOLO 형식에서는 클래스 ID가 0부터 시작해야 함
                    if category_id < 0:
                        continue

                    bbox = ann['bbox']
                    if bbox[2] <= 0 or bbox[3] <= 0:  # 너비나 높이가 0 이하인 경우 필터링
                        continue

                    # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height

                    # YOLO 형식으로 라벨 파일 작성
                    label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

    print(f"YOLO 형식으로 변환된 라벨이 {image_dir}에 저장되었습니다.")