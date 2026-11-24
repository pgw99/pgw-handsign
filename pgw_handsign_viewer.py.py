"""
pgw_handsign_viewer.py
======================
[1단계] 웹캠 손동작 뷰어

USB/CSI 카메라로 들어오는 화면에서 손을 찾아내고,
가위·바위·보 중 무엇인지와 그 확신도(%)를 사각형과 함께 겹쳐 그린다.
화면 좌측 상단에는 초당 처리 장수(FPS)를 띄운다.  'q' 를 누르면 끝난다.
"""

import os
import time
from collections import namedtuple

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter      # 라즈베리파이용 경량 런타임
except ImportError:
    from tensorflow.lite import Interpreter                 # PC 확인용


# ── 고정값 ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(_HERE), "models", "pgw_handsign_detector.tflite"),
    os.path.join(_HERE, "pgw_handsign_detector.tflite"),
]

# 검출기 출력 번호(+1 보정 후) → (표시 이름, 박스 색 BGR)
GESTURES = {
    1: ("Scissors", (255, 0, 0)),
    2: ("Rock", (0, 255, 0)),
    3: ("Paper", (0, 0, 255)),
}

CAPTURE_W, CAPTURE_H = 320, 240
NET_SIDE = 320
MIN_CONF = 0.8                 # 이 값을 넘어야 화면에 표시

Hand = namedtuple("Hand", "gid conf left top right bottom")


# ── 준비 ─────────────────────────────────────────────────────────
def locate_model():
    for candidate in MODEL_SEARCH_PATHS:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("검출 모델 파일이 없습니다:\n  " + "\n  ".join(MODEL_SEARCH_PATHS))


def open_camera():
    cam = cv2.VideoCapture(0)
    for prop, val in ((cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W),
                      (cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H),
                      (cv2.CAP_PROP_BUFFERSIZE, 1)):          # 지연 최소화
        cam.set(prop, val)
    return cam


def frames(cam):
    """카메라가 살아 있는 동안 한 장씩 넘겨주는 제너레이터"""
    while cam.isOpened():
        ok, shot = cam.read()
        if not ok:
            return
        yield shot


# ── 추론 ─────────────────────────────────────────────────────────
def to_tensor(bgr, dtype):
    """카메라 영상 → 네트워크 입력 (1 x 320 x 320 x 3, RGB, -1~1)"""
    rgb = cv2.resize(bgr, (NET_SIDE, NET_SIDE))[..., ::-1]
    scaled = rgb / 127.5 - 1.0
    return scaled[np.newaxis].astype(dtype)


def read_hands(net, out_info, width, height):
    """네트워크 출력에서 기준을 넘은 손만 골라 픽셀 좌표로 바꾼다."""
    fetch = lambda k: net.get_tensor(out_info[k]["index"])[0]
    conf, rect, kind = fetch(0), fetch(1), fetch(3)

    hands = []
    for k in np.flatnonzero(conf > MIN_CONF):
        gid = int(kind[k]) + 1
        if gid not in GESTURES:
            continue
        y0, x0, y1, x1 = rect[k]                     # 0~1 정규화 좌표
        hands.append(Hand(gid, float(conf[k]),
                          int(x0 * width), int(y0 * height),
                          int(x1 * width), int(y1 * height)))
    return hands


# ── 표시 ─────────────────────────────────────────────────────────
def paint(canvas, hands, fps):
    for h in hands:
        name, bgr = GESTURES[h.gid]
        cv2.rectangle(canvas, (h.left, h.top), (h.right, h.bottom), bgr, 2)
        caption = f"{name}: {round(h.conf * 100)}%".upper()
        cv2.putText(canvas, caption, (h.left, max(20, h.top - 7)),
                    cv2.FONT_HERSHEY_PLAIN, 1, bgr, 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)


# ── 실행 ─────────────────────────────────────────────────────────
def launch():
    weights = locate_model()
    net = Interpreter(model_path=weights)
    net.allocate_tensors()
    in_info = net.get_input_details()
    out_info = net.get_output_details()

    print("model   :", weights)
    print("input   :", in_info)
    print("outputs :", out_info)

    cam = open_camera()
    prev = time.time()
    for shot in frames(cam):
        net.set_tensor(in_info[0]["index"], to_tensor(shot, in_info[0]["dtype"]))
        net.invoke()

        hands = read_hands(net, out_info, shot.shape[1], shot.shape[0])

        now = time.time()
        fps, prev = 1.0 / (now - prev), now
        paint(shot, hands, fps)

        cv2.imshow("PGW Hand Gesture Viewer", shot)
        if cv2.waitKey(10) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    launch()
