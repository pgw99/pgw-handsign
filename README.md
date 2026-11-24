# pgw-handsign

라즈베리파이 + 카메라로 손동작(가위 · 바위 · 보)을 실시간 인식하는 프로젝트

## 진행 단계

| 단계 | 내용 | 파일 | 상태 |
|---|---|---|---|
| 1단계 | 카메라 영상에서 손동작 검출 및 표시 | `pgw_handsign_viewer.py` | 완료 |
| 2단계 | 검출 모델 경량화 (FP32 → INT8 양자화) | `pgw_handsign_compressor.py` | 완료 |
| 2단계 | 인식 품질 개선 (입력 영상 보정) | `pgw_handsign_tuner.py` | 진행 중 |
| 최종 | 두 손 대결 승패 판정 | - | 예정 |

## 파일 설명

### `pgw_handsign_viewer.py` — 1단계 손동작 뷰어
- 카메라(320x240) 영상을 TFLite SSD 검출 모델에 넣어 손 위치와 종류를 찾음
- 확신도 80% 이상인 손만 박스와 이름, 확신도로 표시, 화면에 FPS 표시

### `pgw_handsign_compressor.py` — 2단계 모델 경량화 (PC에서 실행)
- Keras 모델(`.h5`)을 TFLite 형식으로 변환
- FP32 버전과 INT8 양자화 버전을 각각 만들어 용량과 입출력 정보를 비교
- INT8 변환 시 보정용 이미지(최대 200장)로 값의 범위를 추정

### `pgw_handsign_tuner.py` — 2단계 인식 개선 작업본
1단계에서 손 모양을 자주 헷갈리는 문제를 확인하고, 입력 영상 처리를 개선하는 중

- [x] 가운데 정사각형 크롭 (화면 비율 왜곡 제거)
- [x] 640x480 고해상도 입력 후 축소
- [x] CLAHE 대비 보정 (어두운 환경 대응)
- [x] 실행 중 확신도 기준 조절 (`+` / `-`, `c`: 보정 ON/OFF)
- [ ] 겹치는 박스 정리
- [ ] 프레임 간 판정 안정화
- [ ] 촬영/추론 분리로 FPS 향상
- [ ] INT8 모델 적용 및 비교

## 실행 방법

```bash
pip install -r requirements.txt

# 라즈베리파이
python3 pgw_handsign_viewer.py
python3 pgw_handsign_tuner.py

# PC (모델 변환, tensorflow 필요)
pip install tensorflow
python pgw_handsign_compressor.py
```

모델 파일은 용량 문제로 저장소에 포함하지 않음.
`pgw_handsign_detector.tflite` 를 코드와 같은 폴더 또는 상위 폴더의 `models/` 에 넣고 실행.

## 실행 환경
- Raspberry Pi + USB/CSI 카메라
- Python 3, tflite-runtime, OpenCV, NumPy
