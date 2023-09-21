from ultralytics import YOLO

def predict_init(args):
    global model
    print("Predict on", args.model_dir + args.model)
    model = YOLO(args.model_dir + args.model)
def predict(args,img):
    global model
    res = model(img, verbose=args.verbos, half=args.half, iou=args.iou, conf=args.conf)
    #print(res)
    return res[0]
