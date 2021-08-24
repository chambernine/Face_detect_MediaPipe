import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    _,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(img,bbox,(255,0,255),2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),1)
    cv2.imshow("Res",img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()