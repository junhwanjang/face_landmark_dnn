from multiprocessing import Process, Queue
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream 
from mark_detector import MarkDetector
import time

from kalman_filter import Stabilizer
from optical_flow_tracker import Tracker
import dlib

CNN_INPUT_SIZE = 64

def webcam_main():
    print("Camera sensor warming up...")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    mark_detector = MarkDetector()
    
    # loop over the frames from the video stream
    while True:
        _, frame = vs.read()
        start = cv2.getTickCount()

        frame = imutils.resize(frame, width=750, height=750)
        frame = cv2.flip(frame, 1)
        faceboxes = mark_detector.extract_cnn_facebox(frame)

        if faceboxes is not None:
            for facebox in faceboxes:
                # Detect landmarks from image of 64X64 with grayscale.
                face_img = frame[facebox[1]: facebox[3],
                                    facebox[0]: facebox[2]]
                # cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)

                land_start_time = time.time()
                marks = mark_detector.detect_marks_keras(face_img0)
                # marks *= 255
                marks *= facebox[2] - facebox[0]
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                # Draw Predicted Landmarks
                mark_detector.draw_marks(frame, marks, color=(255, 255, 255), thick=2)

        fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        cv2.putText(frame, '%.1ffps'%(1/fps_time) , (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        # show the frame
        cv2.imshow("Frame", frame)
        # writer.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    webcam_main()