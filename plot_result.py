import os
import numpy as np
import json
from PIL import Image, ImageDraw
import cv2
import shutil
import argparse
from src.config import ConfigYOLOV3DarkNet53

def parse_args():
    parser = argparse.ArgumentParser('saving predictions')
    parser.add_argument('--bbox_txt', type=str, default='./outputs/2021-06-02_time_09_34_13/predict_2021_06_02_09_35_17.json', help='predictions of eval.py')
    parser.add_argument('--img_path', type=str, default='./fake_images/fake_pulsar_test_2/',
                        help='test image path')
    parser.add_argument('--save_path', type=str, default='./fake_images/fake_pulsar_test_2_prediction/',
                        help='save path')
    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    bbox_txt = args.bbox_txt
    img_path = args.img_path
    img_list = os.listdir(img_path)
    predictions = json.load(open(bbox_txt, 'r'))
    save_path = args.save_path
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.mkdir(save_path)
    config = ConfigYOLOV3DarkNet53()
    test_image_size = config.test_img_shape

    for bbox in predictions:
        image_name = bbox['image_name']
        name_split = image_name.split('.png')[0].split('_')
        img_id = int(float(name_split[-1]))
        for item in img_list:
            if '_' + str(img_id) + '.png' in item:
                break
        # add bboxing on image and save it
        img = cv2.imread(img_path + item)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(test_image_size[1]), int(test_image_size[0])))
        for bbox_cord in bbox['bbox']:
            top_left = (int(bbox_cord[0]), int(bbox_cord[1]))
            bot_right = (int(bbox_cord[2]), int(bbox_cord[3]))
            cv2.rectangle(img, top_left, bot_right, (0, 255, 0), 2)
        # cv2.imshow('detection', img)
        cv2.imwrite(save_path + item, img)
        print("saving to {}...".format(save_path + item))

    print("finish...")

