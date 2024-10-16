import argparse
import os
import torch
import pandas as pd
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device

def detect_and_save_csv(weights='weights/best.pt', source='data/images', img_size=640, conf_thres=0.25, iou_thres=0.45, save_csv_path='output.csv'):
    # Initialize
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    # Load dataset
    dataset = LoadImages(source, img_size=img_size)

    # Run inference
    results = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results to list
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / im0.shape[1::-1]).view(-1).tolist()
                    results.append([p, cls.item(), conf.item(), *xywh])

    # Save to CSV
    results_df = pd.DataFrame(results, columns=['filename', 'class', 'confidence', 'x_center', 'y_center', 'width', 'height'])
    results_df.to_csv(save_csv_path, index=False)
    print(f'Results saved to {save_csv_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-csv-path', type=str, default='output.csv', help='path to save CSV file')
    opt = parser.parse_args()

    detect_and_save_csv(opt.weights, opt.source, opt.img_size, opt.conf_thres, opt.iou_thres, opt.save_csv_path)