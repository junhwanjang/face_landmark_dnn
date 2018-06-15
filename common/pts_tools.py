import glob
import cv2
import numpy as np
import os
import json

def read_points(pts_path):
    with open(pts_path) as file:
        landmarks = []
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                x, y = line.strip().split(" ")
                landmarks.append([float(x), float(y)])
        landmarks = np.array(landmarks)
    return landmarks

def get_paths(dir, dataset_name):
    if dataset_name == "300W":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw1":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw2":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "300VW":
        img_paths = glob.glob(dir + dataset_name + "/*/annot/*.jpg")
    else:
        img_paths = glob.glob(dir + dataset_name + "/*.jpg")
    return img_paths