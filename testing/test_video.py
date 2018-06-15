from multiprocessing import Process, Queue
import numpy as np
import cv2
from mark_detector import MarkDetector
import time

TENSOR_Based = False
PLAY_VIDEO = True
CNN_INPUT_SIZE = 64

# Model File Path
LANDMARK_MODEL = "../landmark_model/Mobilenet_v1.hdf5"
# LANDMARK_MODEL = "../assets/frozen_inference_graph.pb" # yinguobing model

# Video File Path
VIDEO_PATH = "../samples/IU.avi"

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def single_main(play_video=PLAY_VIDEO):
    """MAIN"""
    if PLAY_VIDEO:
        cam = cv2.VideoCapture(VIDEO_PATH)
        time.sleep(2.0)
        _, sample_frame = cam.read()
        # Load Landmark detector
        mark_detector = MarkDetector(LANDMARK_MODEL)

        # Setup process and queues for multiprocessing
        img_queue = Queue()
        box_queue = Queue()
        img_queue.put(sample_frame)
        box_process = Process(target=get_face, args=(
            mark_detector, img_queue, box_queue,))
        box_process.start()

        if TENSOR_Based:
            while True:
                frame_got, frame = cam.read()
                if frame_got is False:
                    break
                img_queue.put(frame)

                # Get face from box queue.
                facebox = box_queue.get()
                if facebox is not None:
                    # Detect landmarks from image of 128x128.
                    face_img = frame[facebox[1]: facebox[3],
                                    facebox[0]: facebox[2]]
                    cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
                    face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    start_time = time.time()
                    marks = mark_detector.detect_marks(face_img)
                    print(time.time() - start_time)
                    marks *= facebox[2] - facebox[0]
                    marks[:, 0] += facebox[0]
                    marks[:, 1] += facebox[1]

                    # Draw Predicted Landmarks
                    mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

                # Show preview.
                cv2.imshow("Preview", frame)
                if cv2.waitKey(10) == ord("q"):
                    break

        else:
            while True:
                frame_got, frame = cam.read()
                if frame_got is False:
                    break
                img_queue.put(frame)

                # Get face from box queue.
                facebox = box_queue.get()

                if facebox is not None:
                    # Detect landmarks from image of 128x128.
                    face_img = frame[facebox[1]: facebox[3],
                                    facebox[0]: facebox[2]]
                    cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
                    face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img0 = face_img.reshape(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 1)
                    start_time = time.time()
                    marks = mark_detector.detect_marks(face_img0)
                    print(time.time() - start_time)
                    marks *= facebox[2] - facebox[0]
                    marks[:, 0] += facebox[0]
                    marks[:, 1] += facebox[1]

                    # Draw Predicted Landmarks
                    mark_detector.draw_marks(frame, marks, color=(0, 255, 0))
                    
                # Show preview.
                cv2.imshow("Preview", frame)
                if cv2.waitKey(10) == ord('q'):
                    break

        # Clean up the multiprocessing process.
        box_process.terminate()
        box_process.join()

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    single_main()
