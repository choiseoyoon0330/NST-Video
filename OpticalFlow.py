import cv2 as cv
import numpy as np

cap = cv.VideoCapture("C:/content_3.mp4")

ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

mask = np.zeros_like(first_frame)   # HSV 이미지. 초기화 시 모든 값을 0으로 설정      
mask[..., 1] = 255       # 색조 채널을 255로 설정하여 채도를 최대로 만든다

threshold = 3.0

frame_index = 0
cnt = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Optical Flow 계산
        # prev_gray: 이전 프레임의 그레이 스케일 이미지
        # gray: 현재 프레임의 그레이 스케일 이미지
        # 나머지 매개변수: Farneback 알고리즘의 파라미터들
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Optical Flow 시각화
        # cv.cartToPolar: Optical Flow의 x, y 벡터를 극좌표로 변환
        # magnitude(이동 크기), angle(각도) 반환
        magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

        if np.mean(magnitude) > threshold:
            name = f'frame_{frame_index:04d}.jpg'
            cv.imwrite(name, frame)
            cnt += 1
            print(f'Frame Saved: {name}')  

        prev_gray = gray     # prev_gray를 현재 프레임의 그레이스케일 이미지로 업데이트
        frame_index += 1

    else:
        break

print(cnt)

cap.release()
cv.destroyAllWindows()