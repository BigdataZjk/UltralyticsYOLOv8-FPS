import win32gui
import win32api
import win32gui
import win32con


def Show_target(box):
    hwnd = win32gui.GetDesktopWindow()
    hPen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 255))  # 定义框颜色

    hwndDC = win32gui.GetDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）

    win32gui.SelectObject(hwndDC, hPen)
    hbrush = win32gui.GetStockObject(win32con.NULL_BRUSH)  # 定义透明画刷，这个很重要！！
    prebrush = win32gui.SelectObject(hwndDC, hbrush)
    win32gui.Rectangle(hwndDC, box[0], box[1], box[2], box[3])  # 左上到右下的坐标
    win32gui.SaveDC(hwndDC)
    win32gui.SelectObject(hwndDC, prebrush)
    win32gui.ReleaseDC(hwnd, hwndDC)
    return
