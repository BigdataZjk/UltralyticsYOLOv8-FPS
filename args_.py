import os


def arg_init(args):
    dirpath = os.path.dirname(os.path.realpath(__file__))
    args.add_argument("--dir", type=str, default=dirpath, help="root dir path")
    args.add_argument(
        "--save_dir", type=str, default=dirpath + "/predict", help="save dir"
    )
    args.add_argument(
        "--model_dir", type=str, default=dirpath + "/model", help="model dir"
    )
    args.add_argument("--model", type=str,
                      default="/apex_1w_yolov8n_fp16.trt", help="model path")
    args.add_argument("--iou", type=float,
                      default=0.8, help="predict iou")
    args.add_argument("--classes",
                      type=int,
                      default=[1,2],
                      help="classes to be detected, can be expanded but needs to be an array. "
                           "For example, the default weight has: "
                           "0 represents 'Teammate',"
                           "1 represents 'Enemy', "
                           "2 represents 'Hitmark'..."
                           "Change default accordingly if your dataset changes")
    args.add_argument("--conf", type=float,
                      default=0.7, help="predict conf")
    args.add_argument("--crop_size", type=float,
                      default=1 / 3,
                      help="the portion to detect from the screen(=crop_window_height/screen_height)"
                           "(It's always a rectangle)(from 0 to 1)")
    args.add_argument("--wait", type=float, default=0, help="wait time")
    args.add_argument("--verbos", type=bool, default=False, help="predict verbos")
    args.add_argument("--target_index", type=int,
                      default=1, help="target index")
    args.add_argument("--half", type=bool, default=True,
                      help="use half to predict")

    # PID args
    args.add_argument("--pid", type=bool, default=True, help="use pid")
    args.add_argument("--Kp", type=float, default=0.8, help="Kp")
    args.add_argument("--Ki", type=float, default=0.05, help="Ki")
    args.add_argument("--Kd", type=float, default=0.1, help="Kd")

    args = args.parse_args(args=[])
    return args
