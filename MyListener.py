from pynput import mouse, keyboard
import numpy as np
import pyautogui
import time
import win32api
import win32con
import win32gui
import win32print
import winsound

Start_detection = False
left_lock = False  # 左键锁, Left, 按鼠标左键时锁
right_lock = False  # 右键锁, Right, 按鼠标右键(开镜)时锁
old_status = False
Listen = True
width = 0
interval = 0.01
screen_size = np.array([win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)])
screen_center = np.array(screen_size, dtype=int) // 2
destination = screen_center
last = destination
backforce = 0
hDC = win32gui.GetDC(0)
scale = win32print.GetDeviceCaps(hDC, win32con.LOGPIXELSX) / 96


def get_S_L():
    global Start_detection
    global Listen
    return Start_detection, Listen


# if the esc is clicked, return True
def listen_key(key):
    if key == keyboard.Key.home:
        global Start_detection, Listen
        Listen = False
        Start_detection = False
        print("Stop listening")
        return False
    if key == keyboard.Key.end:
        Start_detection = not Start_detection
        print("Start detection: ", Start_detection)
    if key == keyboard.Key.shift:
        Start_detection = True
        # print("Start detection: ", Start_detection)
    if key == keyboard.Key.left:
        global left_lock
        left_lock = not left_lock
        winsound.Beep(800 if left_lock else 400, 200)
    if key == keyboard.Key.right:
        global right_lock
        right_lock = not right_lock
        winsound.Beep(800 if right_lock else 400, 200)


# TEMPORARY - detect if key shift is released
def keys_released(key):
    if key == keyboard.Key.shift:
        global Start_detection
        Start_detection = False
        # print("Start detection: ", Start_detection)


# if the right mouse is clicked, return True
def listen_mouse(x, y, button, pressed):
    global Start_detection, old_status, backforce, left_lock, right_lock
    if button == mouse.Button.right and right_lock:
        if pressed:
            Start_detection = True
            print("Start detection: ", Start_detection)
        if not pressed:
            Start_detection = False
            print("Start detection: ", Start_detection)
    if button == mouse.Button.left and left_lock:
        if pressed:
            Start_detection = True
            backforce = 3
            print("Start detection: ", Start_detection)
        if not pressed:
            Start_detection = False
            backforce = 0
            print("Start detection: ", Start_detection)


def speed_func(x, speed, smooth):
    global width
    return max(x / smooth / speed, width / 3)


pre_error = intergral = np.array([0., 0.])


def PID(args, error):
    global pre_error, intergral, backforce
    intergral += error
    derivative = error - pre_error
    pre_error = error
    output = args.Kp * error + args.Ki * intergral + args.Kd * derivative
    output[1] += backforce
    pre_error = error
    return output.astype(int)


def Move_Mouse(args):
    global screen_size, screen_center, last
    # while Listen:
    global destination, width, interval, pre_error, intergral
    if Start_detection:
        pos = np.array(win32api.GetCursorPos(), dtype=int)
        mouse_vector = (destination - pos) / scale
        norm = np.linalg.norm(mouse_vector)
        if args.pid:
            if (destination[0] == -1 and destination[1] == -1):
                if last[0] == -1:
                    pre_error = intergral = np.array([0., 0.])
                    mouse_vector = np.array([0, 0])
                    return
                else:
                    mouse_vector = np.array([0, 0])
            move = PID(args, mouse_vector)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(move[0]), int(move[1]))
            return
        # if destination not in region
        if norm <= 2 or (destination[0] == screen_center[0] and destination[1] == screen_center[1]): return
        if norm <= width * 2 / 3:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_vector[0] / 2), int(mouse_vector[1] / 2))
            return
        des = mouse_vector / args.smooth
        for i in range(int(args.smooth)):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(des[0]), int(des[1]))
            time.sleep(0.01 / args.smooth)
        delay_time = 2.4 / float(args.game_fps)
        time.sleep(delay_time)
    else:
        pre_error = intergral = np.array([0., 0.])


# redirect the mouse closer to the nearest box center
def Mouse_redirection(boxes, args, tpf):
    global destination, width, interval, screen_size, screen_center, last
    if boxes.shape[0] == 0:
        last = destination
        destination = np.array([-1, -1])
        return
    interval = tpf
    pos = np.array(win32api.GetCursorPos(), dtype=int)

    # Get the center of the boxes
    boxes_center = (
            (boxes[:, :2] + boxes[:, 2:]) / 2
    )
    boxes_center[:, 1] = (
            boxes[:, 1] * 0.6 + boxes[:, 3] * 0.4
    )
    # Map the box from the image coordinate to the screen coordinate
    screen_center = screen_size / 2
    start_point = screen_center - screen_size[1] * args.crop_size / 2
    start_point = list(map(int, start_point))
    boxes_center[:, 0] = boxes_center[:, 0] + start_point[0]
    boxes_center[:, 1] = boxes_center[:, 1] + start_point[1]

    # Find the nearest box center
    dis = np.linalg.norm(boxes_center - pos, axis=-1)
    min_index = np.argmin(dis)
    width = boxes[min_index, 2] - boxes[min_index, 0]
    last = destination
    destination = boxes_center[np.argmin(dis)].astype(int)
    # print(destination)

    # mouse_instance.position = tuple(boxes_center[np.argmin(dis)])
