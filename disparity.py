# This python script should create a disparity map

# pip install matplotlib
# pip install numpy
# pip install opencv-contrib-python

# import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np
from point_cloud import PointCloudGen


class DisparityMap:
    def __init__(self, left_path, right_path, calib_path):
        self.left_img = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
        self.right_img = cv.imread(right_path, cv.IMREAD_GRAYSCALE)
        self.calib_path = calib_path

    def compute_depth_SGBM(self):

        # parameters were found through trial and error
        window = 7
        min_disparity = 0
        num_disparity_factor = 14
        num_disparities = 16 * num_disparity_factor - min_disparity

        stereo = cv.StereoSGBM.create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=window,
            P1=8 * window ** 2,
            P2=32 * window ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(self.left_img, self.right_img).astype(np.float32) / 16.0

        # visualization for development
        '''
        plt.imshow(disparity, cmap='gray')
        plt.axis('off')
        plt.show()
        '''

        return disparity


if __name__ == "__main__":
    # paths to images, calibration, and output file
    # replace with any set of images and corresponding calibration file

    root = os.getcwd()
    left_img_path = os.path.join(root, 'data/chair_perfect/im0.png')
    right_img_path = os.path.join(root, 'data/chair_perfect/im1.png')
    calibration_path = os.path.join(root, 'data/chair_perfect/calib.txt')
    output_ply_path = os.path.join(root, 'final_point_cloud.ply')

    # compute and display disparity map
    dm = DisparityMap(left_img_path, right_img_path, calibration_path)
    disp = dm.compute_depth_SGBM()

    # revert image from greyscale back to color and generate the point cloud from that color image + disparities
    left_color = cv.imread(left_img_path)
    colors = left_color[disp > 0]
    pcg = PointCloudGen(calibration_path, disp)
    points = pcg.generate_3D_point_cloud()

    # save the point cloud to visualize
    pcg.save_to_ply(output_ply_path, colors)



