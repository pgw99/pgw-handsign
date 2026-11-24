"""
pgw_handsign_tuner.py
=====================
[2단계 · 진행 중] 손동작 뷰어 인식 품질 개선 작업본

1단계 뷰어에서 "손 모양을 자주 헷갈린다"는 문제를 확인했고,
모델을 다시 학습하기 전에 입력 영상 쪽을 손봐서 얼마나 좋아지는지 실험하는 중이다.

현재 반영된 것
  [O] 비율 유지   : 4:3 화면의 가운데 정사각형만 잘라 넣어 손이 옆으로 눌리지 않게 함
  [O] 해상도 상향 : 640x480 으로 받아 면적 평균(INTER_AREA)으로 축소
  [O] 대비 보정   : 밝기 채널에 CLAHE 적용 (어두운 곳 / 역광 대응)
  [O] 기준값 조절 : 실행 중 키보드로 확신도 기준 변경

다음에 할 것 (TODO)
  [ ] 한 손에 박스가 여러 개 겹쳐 뜨는 현상 정리
  [ ] 판정이 프레임마다 깜빡이는 현상 완화 (최근 몇 장 결과로 다수결)
  [ ] 촬영과 추론을 분리해 FPS 끌어올리기
  [ ] 2단계에서 만든 int8 모델로 바꿔 끼워 속도/정확도 비교

키 조작:  [+]/[-] 확신도 기준 ±0.05   [c] 대비 보정 ON/OFF   [q] 종료
"""

import os
import time

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


# ── 실험 변수 ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FILE_OPTIONS = [
    os.path.join(_HERE, "pgw_handsign_detector.tflite"),
    os.path.join(os.path.dirname(_HERE), "models", "pgw_handsign_detector.tflite"),
]

SIGNS = {1: ("Scissors", (255, 0, 0)), 2: ("Rock", (0, 255, 0)), 3: ("Paper", (0, 0, 255))}

FRAME_W, FRAME_H = 640, 480
NET_SIDE = 320
INITIAL_CUT = 0.6        # 1단계(0.8)보다 낮춰서 시작
SQUARE_ROI = True
USE_CONTRAST = True


# ── 전처리 ───────────────────────────────────────────────────────
def square_window(img):
    """가운데 정사각형 영역 (x, y, 한 변)"""
    rows, cols = img.shape[:2]
    if not SQUARE_ROI:
        return 0, 0, cols, rows
    edge = min(rows, cols)
    return (cols - edge) // 2, (rows - edge) // 2, edge, edge


EQUALIZER = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def brighten(img):
    """LAB 색공간의 밝기(L)만 국소 대비 보정"""
    lum, ca, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((EQUALIZER.apply(lum), ca, cb)), cv2.COLOR_LAB2BGR)


def prepare(patch, dtype, contrast):
    small = cv2.resize(patch, (NET_SIDE, NET_SIDE), interpolation=cv2.INTER_AREA)
    if contrast:
        small = brighten(small)
    signal = small[..., ::-1] / 127.5 - 1.0        # BGR→RGB, -1~1
    return signal[np.newaxis].astype(dtype)
    # TODO: int8 모델은 scale / zero_point 를 이용한 변환이 필요함


# ── 검출 ─────────────────────────────────────────────────────────
def detect(net, feed, taps, img, cut, contrast):
    x, y, w, h = square_window(img)
    net.set_tensor(feed["index"], prepare(img[y:y + h, x:x + w], feed["dtype"], contrast))
    net.invoke()

    pull = lambda k: net.get_tensor(taps[k]["index"])[0]
    conf, rect, kind = pull(0), pull(1), pull(3)

    hits = []
    for c, r, k in zip(conf, rect, kind):
        sign = int(k) + 1
        if c < cut or sign not in SIGNS:
            continue
        t, l, b, rt = np.clip(r, 0.0, 1.0)
        # 잘라낸 영역 기준 좌표를 전체 화면 좌표로 되돌림
        hits.append((sign, float(c),
                     (int(x + l * w), int(y + t * h)),
                     (int(x + rt * w), int(y + b * h))))
    # TODO: 여기서 겹치는 박스 정리 예정
    return hits


# ── 화면 ─────────────────────────────────────────────────────────
def overlay(img, hits, fps, cut, contrast):
    if SQUARE_ROI:
        x, y, w, h = square_window(img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)

    for sign, c, p1, p2 in hits:
        name, tint = SIGNS[sign]
        cv2.rectangle(img, p1, p2, tint, 2)
        cv2.putText(img, f"{name}: {round(c * 100)}%".upper(), (p1[0], max(20, p1[1] - 7)),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, tint, 2)

    status = f"FPS {fps:.1f} | TH {cut:.2f} | CLAHE {'ON' if contrast else 'OFF'}"
    cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 255), 2)


# ── 실행 ─────────────────────────────────────────────────────────
def launch():
    path = next((p for p in WEIGHT_FILE_OPTIONS if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError("검출 모델이 없습니다:\n  " + "\n  ".join(WEIGHT_FILE_OPTIONS))

    net = Interpreter(model_path=path)
    net.allocate_tensors()
    feed, taps = net.get_input_details()[0], net.get_output_details()

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cut, contrast = INITIAL_CUT, USE_CONTRAST
    prev = time.time()
    while cam.isOpened():
        ok, img = cam.read()
        if not ok:
            break

        hits = detect(net, feed, taps, img, cut, contrast)

        now = time.time()
        fps, prev = 1.0 / max(now - prev, 1e-6), now
        overlay(img, hits, fps, cut, contrast)
        cv2.imshow("PGW HandSign Tuner (WIP)", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            cut = min(0.95, cut + 0.05)
        elif key == ord("-"):
            cut = max(0.30, cut - 0.05)
        elif key == ord("c"):
            contrast = not contrast

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    launch()
