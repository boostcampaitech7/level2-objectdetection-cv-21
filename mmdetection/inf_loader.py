import re
import csv

def inf_loader(path) :
    f = open(path, 'r')  #읽기모드로 csv file open

    rdr = csv.reader(f)
    for line in rdr:            #csv file을 한줄씩 리스트형식([])으로 읽어들인다..
        nested_list = line[0]

        flattened_list = []
        for sublist in nested_list:
            flattened_list.extend(sublist)

        print(flattened_list)

# 사용 예시
input_file_path = "submission_faster_rcnn_epoch_30.csv"  # CSV 파일 경로를 여기에 지정
inf_loader(input_file_path)
