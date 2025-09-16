import numpy as np
import cv2


class PointCloudGen:
    def __init__(self, calib_path, disparity_array):
        self.calib = self.read_calib(calib_path)
        self.disparity = disparity_array
        self.Q = self.Q_mat()

    # parse calib.txt and extract useful information for the Q matrix
    def read_calib(self, path):
        calib_params = {}
        with open(path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                if 'cam' in key:
                    matrix = value.replace('[', '').replace(']', '').split(';')
                    calib_params[key] = np.array([[float(num) for num in row.split()] for row in matrix])
                else:
                    calib_params[key] = float(value)

        # debugging step to make sure the array is of the right type; Expected: ndarray
        print("cam0:", calib_params['cam0'], type(calib_params['cam0']))
        return calib_params

    # define Q matrix
    def Q_mat(self):
        cam0 = self.calib['cam0']
        cam1 = self.calib['cam1']
        doffs = self.calib['doffs']
        baseline = self.calib['baseline']

        f = cam0[0, 0]
        cx = cam0[0, 2]
        cy = cam0[1, 2]

        Q_matrix = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1 / baseline, (cx - doffs) / baseline]
        ])
        return Q_matrix

    # generates the point cloud using the Q matrix and disparity values
    def generate_3D_point_cloud(self):
        points_3d = cv2.reprojectImageTo3D(self.disparity, self.Q)
        mask = self.disparity > 0
        points = points_3d[mask]
        return points

    def save_to_ply(self, filename, colors=None):
        points = self.generate_3D_point_cloud()
        if colors is None:
            colors = np.zeros_like(points)

        with open(filename, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for p, c in zip(points, colors):
                f.write(f'{p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]}\n')
