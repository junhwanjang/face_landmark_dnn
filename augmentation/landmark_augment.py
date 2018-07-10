import numpy as np
import cv2
from landmark_helper import LandmarkHelper

class LandmarkAugment(object):
    """
    Facial landmarks augmentation.
    """

    def __init__(self):
        pass

    def augment(self, image, landmarks, output_size, max_angle, scale_range):
        """
        Do image augment.
        Args:
            image: a numpy type
            landmarks: face landmarks with format numpy [(x1, y1), (x2, y2), ..]
            output_size: target image size with format (w, h)
            max_angle: random to rotate in [-max_angle, max_angle]. range is 0-180.
            scale_range: scale bbox in (min, max). eg: (13.0, 15.0)
        Return:
             an image with target size will be return
        Raises:
            No
        """
        image, landmarks = self.__flip(image, landmarks)
        image, landmarks = self.__rotate(image, landmarks, max_angle)
        image, landmarks = self.__scale_and_shift(image, landmarks, scale_range, output_size)
        landmarks.flatten()
        return image, landmarks

    def mini_crop_by_landmarks(self, sample_list, pad_rate, img_format):
        """
        Crop full image to mini. Only keep valid image to save
        Args:
            sample_list: (image, landmarks)
            pad_rate: up scale rate
            img_format: "RGB" or "BGR"
        Return:
            new sample list
        Raises:
            No
        """
        new_sample_list = []
        for sample in sample_list:
            image = cv2.imread(sample[0])
            if img_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = sample[1]
            (x1, y1, x2, y2), _, _, _ = self.get_bbox_of_landmarks(image, landmarks, pad_rate, 0.5)
            new_sample_list.append(
                (cv2.imencode(".jpg", image[y1:y2, x1:x2])[1], landmarks - (x1, y1))
            )
            return new_sample_list

    def __flip(self, image, landmarks, run_prob=0.5):
        """
        Do image flop. Only for horizontal
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), (x2, y2), ...]
            run_prob: probability to do this operate. 0.0-1.0
        Return:
            an image and landmarks will be returned
        Raises:
            Unsupport count of landmarks
        """
        if np.random.rand() < run_prob:
            return image, landmarks
        image = np.fliplr(image)
        landmarks[:, 0] = image.shape[1] - landmarks[:, 0]
        landmarks = LandmarkHelper.flip(landmarks, landmarks.shape[0])
        return image, landmarks

    def __rotate(self, image, landmarks, max_angle):
        """
        Do image rotate
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            max_angle: random to rotate in [-max_angle, max_angle]. range is 0-180.
        Return:
            an image and landmarks will be returned
        Raises:
            No
        """
        c_x = (min(landmarks[:, 0]) + max(landmarks[:, 0])) / 2
        c_y = (min(landmarks[:, 1]) + max(landmarks[:, 1])) / 2
        h, w = image.shape[:2]
        angle = np.random.randint(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        b = np.ones((landmarks.shape[0], 1))
        d = np.concatenate((landmarks, b), axis=1)
        landmarks = np.dot(d, np.transpose(M))
        return image, landmarks

    def __scale_and_shift(self, image, landmarks, scale_range, output_size):
        """
        Auto generate bbox and then random to scale and shift it.
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
            output_size: output size of image
        Return:
            an image and landmarks will be returned
        Raises:
            No
        """
        (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h) = self.get_bbox_of_landmarks(
            image, landmarks, scale_range, shift_rate=0.3)
        box_image = image[y1:y2, x1:x2]
        if need_pad:
            box_image = np.lib.pad(box_image, ((p_y, p_h), (p_x, p_w), (0, 0)), 'constant')
        box_image = cv2.resize(box_image, (output_size, output_size))
        landmarks = (landmarks - (x1 - p_x, y1 - p_y)) / (new_size, new_size)
        return box_image, landmarks

    def get_bbox_of_landmarks(self, image, landmarks, scale_range, shift_rate=0.3):
        """
        According to landmark box to generate a new bigger bbox
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
            shift_rate: up, down, left, right to shift
        Return:
             return new bbox and other info
        Raises:
            No
        """
        ori_h, ori_w = image.shape[:2]
        x = int(min(landmarks[:, 0]))
        y = int(min(landmarks[:, 1]))
        w = int(max(landmarks[:, 0]) - x)
        h = int(max(landmarks[:, 1]) - y)
        if type(scale_range) == float:
            scale = scale_range
        else:
            scale = np.random.randint(int(scale_range[0] * 100.0), int(scale_range[1] * 100.0)) / 100.0
        new_size = int(max(w, h) * scale)
        if shift_rate >= 0.5:
            x1 = x - (new_size - w) / 2
            y1 = y - (new_size - h) / 2
        else:
            x1 = x - np.random.randint(int((new_size - w) * shift_rate), int((new_size - w) * (1.0 - shift_rate)))
            y1 = y - np.random.randint(int((new_size - w) * shift_rate), int((new_size - w) * (1.0 - shift_rate)))
        x2 = x1 + new_size
        y2 = y1 + new_size
        need_pad = False
        p_x, p_y, p_w, p_h = 0, 0, 0, 0
        if x1 < 0:
            p_x = -x1
            x1 = 0
            need_pad = True
        if y1 < 0:
            p_y = -y1
            y1 = 0
            need_pad = True
        if x2 > ori_w:
            p_w = x2 - ori_w
            x2 = ori_w
            need_pad = True
        if y2 > ori_h:
            p_h = y2 - ori_h
            y2 = ori_h
            need_pad = True

        return (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h)


