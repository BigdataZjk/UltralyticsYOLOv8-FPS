from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 2  # your model classes

def predict_init(args):
    global pred
    pred = Predictor(engine_path=args.model_dir + args.model)

def predict_trt(args, img):
    if img is not None:
      boxes = pred.inference(img, conf=args.conf, classes=args.classes, end2end=True)
    return boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")

    args = parser.parse_args()
    print(args)
