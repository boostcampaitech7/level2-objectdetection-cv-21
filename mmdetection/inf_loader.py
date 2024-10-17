import re
import csv

def convert_string_to_lists(input_string):
    # 정규표현식을 사용하여 대괄호 안의 숫자들을 추출
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, input_string)

    # 추출된 문자열을 리스트로 변환
    result = []
    for match in matches:
        # 쉼표로 구분된 숫자들을 실수로 변환
        numbers = [float(num.strip()) for num in match.split(',')]
        result.append(numbers)

    return result

def process_csv_file(input_file_path):
    try:
        with open(input_file_path, 'r') as file:
            csv_reader = csv.reader(file)

            # 각 줄을 처리
            for row in csv_reader:
                if len(row) >= 2:  # 최소한 2개의 열이 있는지 확인
                    input_string = row[0]  # 첫 번째 열의 문자열
                    filename = row[1]      # 두 번째 열의 파일명

                    # 문자열 변환 수행
                    result_lists = convert_string_to_lists(input_string)

                    # 결과 출력
                    print(f"\nProcessing {filename}:")
                    for sublist in result_lists:
                        print(sublist)

    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# 사용 예시
input_file_path = "submission_faster_rcnn_epoch_30.csv"  # CSV 파일 경로를 여기에 지정
process_csv_file(input_file_path)
