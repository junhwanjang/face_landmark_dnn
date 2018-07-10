import glob
import cv2
import numpy as np
import os
import json

import sys
sys.path.append("../utils/")
from pts_tools import read_points, get_paths
from landmark_augment import LandmarkAugment

DIR = "../samples/"
OUTDIR = "../samples/out/"

# Initialize 
aug = LandmarkAugment()

def main():
    img_paths = get_paths(DIR, dataset_name="300W")

    for idx, img_path in enumerate(img_paths):
        image = cv2.imread(img_path)
        pts_path = img_path.split(".")[0] + ".txt"
        landmarks = read_points(pts_path)
        image_aug, landmarks_aug = aug.augment(image=image, landmarks=landmarks, output_size=64,
                                    max_angle=5, scale_range=1.3)

        is_valid_points = len(landmarks_aug) == 68
        gray = cv2.cvtColor(image_aug, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        if is_valid_points:
            cv2.imwrite(OUTDIR + str(idx) + ".jpg", gray)
            np.savetxt(output_folder + str(idx) + ".pts", landmarks_aug)
            print(idx)

if __name__ == "__main__":
    main()