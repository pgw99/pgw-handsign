"""
pgw_handsign_compressor.py
==========================
[2단계] 손동작 검출 네트워크 경량화 도구

학습이 끝난 Keras 가중치 파일(.h5)을 엣지 기기에서 쓸 수 있는
TensorFlow Lite 플랫버퍼로 내보낸다.

  * float32 그래프 : 정밀도 손실 없는 기준본 (비교용)
  * int8 그래프    : 보정용 사진으로 활성값 분포를 추정해 8비트 정수로 압축한 배포본

실행이 끝나면 두 그래프의 텐서 정보와 용량 차이를 표 형태로 보여준다.
TensorFlow 2.13 이상이 설치된 PC 에서 실행하는 것을 전제로 한다.
"""

import os

import cv2
import numpy as np
import tensorflow as tf


# ── 위치 정보 ────────────────────────────────────────────────────
# 이 스크립트가 들어있는 폴더의 한 단계 위를 프로젝트 홈으로 본다.
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FOLDER = os.path.join(PROJECT_HOME, "models")
CALIBRATION_FOLDER = os.path.join(PROJECT_HOME, "assets", "representative_images")

ARTIFACTS = {
    "h5": os.path.join(WEIGHTS_FOLDER, "pgw_handsign_ssdlite.h5"),
    "savedmodel": os.path.join(WEIGHTS_FOLDER, "pgw_handsign_savedmodel"),
    "float32": os.path.join(WEIGHTS_FOLDER, "pgw_handsign_fp32.tflite"),
    "int8": os.path.join(WEIGHTS_FOLDER, "pgw_handsign_int8.tflite"),
}

NET_SIDE = 64          # 네트워크가 받는 정사각형 한 변 길이(px)
CALIB_LIMIT = 200      # 활성값 범위 추정에 쓰는 사진 장수 상한


# ── 보정용 사진 공급 ─────────────────────────────────────────────
def list_calibration_pictures():
    """보정 폴더 안의 png 파일 경로를 이름순으로 정렬해 돌려준다."""
    if not os.path.isdir(CALIBRATION_FOLDER):
        return []
    names = sorted(n for n in os.listdir(CALIBRATION_FOLDER) if n.endswith(".png"))
    return [os.path.join(CALIBRATION_FOLDER, n) for n in names]


def calibration_stream():
    """
    변환기가 한 장씩 꺼내 쓰는 제너레이터.
    사진을 네트워크 입력 규격(1 x 64 x 64 x 3, RGB, 0~1 실수)으로 맞춰 내보낸다.
    """
    pictures = list_calibration_pictures()[:CALIB_LIMIT]
    if not pictures:
        raise FileNotFoundError(f"보정용 png 사진이 한 장도 없습니다 → {CALIBRATION_FOLDER}")

    for picture in pictures:
        pixels = cv2.imread(picture)
        if pixels is None:          # 깨진 파일은 건너뜀
            continue
        pixels = cv2.resize(pixels, (NET_SIDE, NET_SIDE))[..., ::-1]   # 축소 후 BGR→RGB
        batch = pixels.astype(np.float32) / 255.0
        yield [batch[np.newaxis]]


# ── 플랫버퍼 생성 ────────────────────────────────────────────────
# 8비트 압축일 때만 변환기에 덧씌우는 옵션 묶음
INT8_OPTIONS = {
    "optimizations": [tf.lite.Optimize.DEFAULT],
    "representative_dataset": calibration_stream,
    "inference_input_type": tf.int8,
    "inference_output_type": tf.int8,
}


def make_flatbuffer(precision):
    """precision('float32' 또는 'int8')에 맞는 TFLite 바이트열을 만든다."""
    conv = tf.lite.TFLiteConverter.from_saved_model(ARTIFACTS["savedmodel"])
    if precision == "int8":
        for option, value in INT8_OPTIONS.items():
            setattr(conv, option, value)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return conv.convert()


# ── 결과 리포트 ──────────────────────────────────────────────────
def tensor_rows(details):
    return [
        (d["name"], d["shape"].tolist(), np.dtype(d["dtype"]).name, d["quantization"])
        for d in details
    ]


def print_report(precision, nbytes):
    probe = tf.lite.Interpreter(model_path=ARTIFACTS[precision])
    probe.allocate_tensors()

    print(f"[{precision:>7}] {ARTIFACTS[precision]}  ({nbytes:,} bytes)")
    for tag, details in (("in ", probe.get_input_details()), ("out", probe.get_output_details())):
        for name, shape, dtype, quant in tensor_rows(details):
            print(f"    {tag} | {name:<40} | {str(shape):<18} | {dtype:<8} | q={quant}")
    print()


def pipeline():
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

    # Keras 가중치 → SavedModel 디렉터리 (변환기의 입력 형식)
    network = tf.keras.models.load_model(ARTIFACTS["h5"], compile=False)
    network.export(ARTIFACTS["savedmodel"])

    # 두 가지 정밀도로 각각 내보내기
    footprint = {}
    for precision in ("float32", "int8"):
        with open(ARTIFACTS[precision], "wb") as fh:
            fh.write(make_flatbuffer(precision))
        footprint[precision] = os.path.getsize(ARTIFACTS[precision])

    print("=" * 90)
    for precision, nbytes in footprint.items():
        print_report(precision, nbytes)

    shrink = 100.0 * (1.0 - footprint["int8"] / footprint["float32"])
    print(f">> int8 배포본은 float32 기준본보다 {shrink:.1f}% 가볍습니다.")
    print("=" * 90)


if __name__ == "__main__":
    pipeline()
