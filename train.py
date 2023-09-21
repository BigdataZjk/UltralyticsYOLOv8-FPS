from ultralytics import YOLO
from args_ import *
import argparse

# Train YOLOv5s on dataset for 100 epochs
def train(args):
    #model = YOLO("yolov8n.pt")
    #model.train(data=args.dir + "/dataset/Apex_total/data.yaml", epochs=300, batch=20, imgsz=(416,416), workers=16, val=False)
    model = YOLO(args.dir+"/model/apex_total_8n_350_1000.pt")
    model.val(data=args.dir + "/dataset/Apex_total/data.yaml")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = arg_init(args)
    train(args)
