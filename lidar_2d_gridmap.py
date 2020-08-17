################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

import my_params


def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.

        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative to camera frame.
            image_size (tuple[int]): dimensions of image in pixels

        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates of a point in pixels.
            numpy.array: array of depth values for points in image.

        Note:
            Number of output points m will be less than or equal to number of input points n, as points that do not
            project into the image are discarded.

        """
        if xyz.shape[0] == 3:       # (4, 34659)
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(self.G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        in_img = [i for i in range(0, uv.shape[1])
                  if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]]

        return uv[:, in_img], np.ravel(xyzw[2, in_img])





image_dir = my_params.image_dir
laser_dir = my_params.laser_dir
poses_file = my_params.poses_file
models_dir = my_params.model_dir
extrinsics_dir = my_params.extrinsics_dir
image_idx = 1572

model = CameraModel(models_dir, image_dir)


extrinsics_path = os.path.join(extrinsics_dir, model.camera + '.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

# print("G_camera_vehicle",G_camera_vehicle)

poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
if poses_type in ['ins', 'rtk']:
    with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

timestamps_path = os.path.join(my_params.dataset_patch + model.camera + '.timestamps')

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if i == image_idx:
            # print('index:', i)
            timestamp = int(line.split(' ')[0])
            break

# print('index timestamp:', timestamp)
# print('start time:', start_time)
# print('end time:', end_time)

pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,
                                           timestamp - 2e6, timestamp + 2e6, timestamp)

# real point cloud
xyz = pointcloud[0:3,:]
print(xyz.shape)

pointcloud = np.dot(G_camera_posesource, pointcloud)


image_path = os.path.join(image_dir, str(timestamp) + '.png')
image = load_image(image_path, model)

# Print filename
# print('image patch: ',image_path)

uv, depth = model.project(pointcloud, image.shape)

# newmap = np.zeros((800,800),dtype=float)
# plt.imshow(newmap)
# np.ravel(uv[0, :]),np.ravel(uv[3, :])
print(uv.shape)

# plt.figure(1)
# plt.imshow(image)
# plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
# plt.xlim(0, image.shape[1])
# plt.ylim(image.shape[0], 0)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# 2D lidar pointcloud image
# plt.figure(2)
# plt.scatter(np.ravel(uv[0, :]),np.ravel(depth[:]), s=2, c=depth, edgecolors='none', cmap='jet')
# plt.show()
for i in range(xyz.shape[1]):
    depth = pointcloud[2,i]
    plt.scatter(pointcloud[0,i],pointcloud[1,i], s=2, c=depth, edgecolors='none', cmap='jet')
plt.show()
