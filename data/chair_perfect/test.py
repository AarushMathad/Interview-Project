import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Define the path to the PLY file

ply_file_path_read = "/Users/AarushMathad/Downloads/Interview-Project/output.ply"


pointcloud = o3d.io.read_point_cloud(ply_file_path_read)
pointcloud_array = np.asarray(pointcloud.points)

print(f"Point cloud details: {np.shape(pointcloud_array)}")

# Visualize the point cloud using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pointcloud_array[:, 0],
           pointcloud_array[:, 1],
           pointcloud_array[:, 2],
           s=3,
           c='blue',
           alpha=1.0)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('Sample Point cloud visualization using Matplotlib')
plt.show()

# Visualize the point cloud using Open3D
o3d.visualization.draw_geometries([pointcloud], window_name='Sample Point cloud visualization using Open3D')