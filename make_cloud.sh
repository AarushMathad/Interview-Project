#!/bin/bash

#
set -e
source venv/bin/activate

pip install numpy
pip install opencv-contrib-python
pip install matplotlib

echo "computing disparity and creating point cloud"

# main driver code is in disparity.py
python disparity.py

echo "code finished, output saved as final_point_cloud.ply"
