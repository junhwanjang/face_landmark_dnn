import cv2
import glob
import numpy as np

PATH = "../samples/out/"

# Create Array using cropped and augmented images
def main():
    img_list = glob.glob(PATH + "/*.jpg")
    Height = 64
    Width = 64
    N_channel = 1

    dataset_array = np.zeros(shape=(len(img_list), Height, Width, N_channel))
    pts_array = np.zeros(shape=(len(img_list), 68, 2))
    for idx, file in enumerate(img_list):
        # Image array
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img = img.reshape(Height, Width, N_channel)
        dataset_array[idx, :, :, :] = img

        # Points array
        pts_path = file.split(".")[0] + ".pts"
        points = []
        pts = open(pts_path)
        for line in pts:
            loc_x, loc_y = line.strip().split(" ")
            points.append([loc_x, loc_y])
        pts_array[idx, :, :] = points
        print(idx)

    print("Create Image array!")
    print("Create Points array!")

    # Save Image and Points array
    np.savez_compressed(PATH + "img_dataset.npz", dataset_array)
    np.savez_compressed(PATH + "pts_dataset.npz", pts_array)
    print("Save Image and Points array")

if __name__ == "__main__":
    main()