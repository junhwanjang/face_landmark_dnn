from multiprocessing import Process, Queue
import numpy as np
import cv2
from mark_detector import MarkDetector
import time
import sys
sys.path.append("../")
TENSOR_Based = False
PLAY_VIDEO = True
CNN_INPUT_SIZE = 64


# Video File Path
VIDEO_PATH = "samples/IU.avi"

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def single_main():
    """MAIN"""
    cam = cv2.VideoCapture(VIDEO_PATH)
    _, sample_frame = cam.read()
    
    # Load Landmark detector
    mark_detector = MarkDetector()
    
    # Setup process and queues for multiprocessing
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    while True:
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        img_queue.put(frame)
        start = cv2.getTickCount()
    
        # Get face from box queue.
        faceboxes = box_queue.get()
        if faceboxes is not None:
            for facebox in faceboxes:
        # if facebox is not None:
                # Detect landmarks from image of 64x64.
                face_img = frame[facebox[1]: facebox[3],
                                facebox[0]: facebox[2]]
                # cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)
                marks = mark_detector.detect_marks_keras(face_img0) # landmark predictor
                marks *= facebox[2] - facebox[0]
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                
                # Draw Predicted Landmarks
                mark_detector.draw_marks(frame, marks, color=(255, 255, 255))
                
        fps_time = (cv2.getTickCount() - start)/cv2.getTickFrequency()
        cv2.putText(frame, '%.1ffps'%(1/fps_time) , (frame.shape[1]-65,15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (127,127,127))
        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == ord('q'):
            break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

    # out.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    single_main()
