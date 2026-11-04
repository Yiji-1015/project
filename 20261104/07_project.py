from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

import csv
import json
from time import perf_counter
Path("outputs").mkdir(exist_ok=True)

def process_image(image):
    """원본 배열 → Gray·Resize·Edge. 파일 읽기는 호출한 쪽에서 합니다."""
    # >>> TODO A07
    # Gray→160×120 Resize→3×3 Gaussian→Canny 80/160을 구현하세요. 반환 변수 gray·small·edge를 만듭니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 원본 크기의 흑백 이미지
    small = cv2.resize(gray, (160, 120))            # 160x120으로 크기를 바꾼 흑백 이미지
    blurred = cv2.GaussianBlur(small, (3, 3), 0)    # 3×3 범위로 흐림 처리하여 노이즈를 완화한다.
    edge = cv2.Canny(blurred, 80, 160)              # 두 임계값 80, 160을 사용해 윤곽선을 검출한다.
    # <<< TODO A07
    return gray, small, edge

def process_frame(frame):
    """Writer가 받을 160×120 BGR 프레임을 반환합니다."""
    # >>> TODO A08
    # process_image(frame)의 세 번째 반환값 edge를 받아 3채널 BGR로 변환해 반환하세요.
    _, _, edge = process_image(frame)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR) # cv2.cvtColor(변환할_이미지, 변환_방법)
    # <<< TODO A08

# 이미지별 처리 결과와 저장 확인 정보를 모을 빈 리스트
rows = []

# 세 이미지 파일을 하나씩 순서대로 처리한다.
for filename in ["scene.png", "noisy.png", "objects.png"]:
    # data 폴더에서 현재 파일을 컬러 이미지로 읽는다.
    image = cv2.imread("data/" + filename)
    # 이미지 처리 시간을 측정하기 위해 시작 시점을 기록한다.
    start = perf_counter(); gray, small, edge = process_image(image); seconds = perf_counter()-start
    stem = Path(filename).stem
    # 세 원본 이미지 각각에 대해 흑백 변환·축소·윤곽선 검출을 수행한다.
    # 각 단계의 결과를 저장하고 저장 정보를 rows에 기록한 뒤,
    # 원본·축소·윤곽선을 나란히 붙인 비교 이미지도 저장한다.
    for stage, result in [("gray", gray), ("small", small), ("edge", edge)]:
        path = f"outputs/p07_{stem}_{stage}.png"; saved = cv2.imwrite(path, result)
        reread = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        rows.append([filename, stage, result.shape[1], result.shape[0], str(result.dtype), saved, np.array_equal(result,reread), seconds])
    preview = cv2.hconcat([cv2.resize(image,(160,120)), cv2.cvtColor(small,cv2.COLOR_GRAY2BGR), cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(f"outputs/p07_{stem}_compare.png", preview)
cap=cv2.VideoCapture("data/sample.avi"); fps=cap.get(cv2.CAP_PROP_FPS)
writer=cv2.VideoWriter("outputs/p07_video.avi", cv2.VideoWriter_fourcc(*"MJPG"),fps,(160,120))
count=0
while cap.isOpened() and writer.isOpened():
    ok, frame=cap.read()
    if not ok: break
    writer.write(process_frame(frame)); count+=1
cap.release();writer.release()
with open("outputs/result_summary.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);w.writerow(["source","stage","width","height","dtype","saved","png_equal","process_s"]);w.writerows(rows)
check=cv2.VideoCapture("outputs/p07_video.avi")
print("이미지 처리표:",len(rows),"행 / 원본3장")
print("Frame 처리/재읽기:",count,int(check.get(cv2.CAP_PROP_FRAME_COUNT)))
print("재읽기 크기/FPS:",int(check.get(3)),int(check.get(4)),check.get(5));check.release()
print("PNG 저장·재읽기 모두 일치:",all(r[5] and r[6] for r in rows))
