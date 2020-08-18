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
import cv2

def spread(xyz):
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
        G_camera_image = []
        intrinsics_path = os.path.join(my_params.model_dir,'stereo_narrow_right.txt')
        with open(intrinsics_path) as intrinsics_file:
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
        G_camera_image = np.array(G_camera_image)

        if xyz.shape[0] == 3:       # (4, 34659)
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        # uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
        #                 self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        # in_img = [i for i in range(0, uv.shape[1])
        #           if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]]

        # return uv[:, in_img], np.ravel(xyzw[2, in_img])
        return xyzw[0, in_front], xyzw[2, in_front]


if __name__ == "__main__":


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


# Project lidar in image
    pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,
                                            timestamp - 1e6, timestamp + 1e6, timestamp)
    pointcloud = np.dot(G_camera_posesource, pointcloud)

    # print('G_camera_posesource',G_camera_posesource)
    # image_path = os.path.join(image_dir, str(timestamp) + '.png')
    # image = load_image(image_path, model)

    # uv, depth = model.project(pointcloud, image.shape)

    # plt.imshow(image)
    # plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
    # plt.xlim(0, image.shape[1])
    # plt.ylim(image.shape[0], 0)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

# 2D lidar image
    # plt.figure(2)
    # plt.scatter(np.ravel(uv[0, :]),np.ravel(depth[:]), s=2, c=depth, edgecolors='none', cmap='jet')
    # plt.show()

    _x = 20*np.ravel(pointcloud[1,:])
    _y = 20*np.ravel(pointcloud[0,:])
    _z = 20*np.ravel(pointcloud[2,:])
    
    grid_min_x = np.min(_x)
    grid_min_y = np.min(_y)
    grid_max_x = np.max(_x)
    grid_max_y = np.max(_y)

    print ('grid_max_x: ', grid_max_x)
    print ('grid_min_x: ', grid_min_x)
    print ('grid_max_y: ', grid_max_y)
    print ('grid_min_y: ', grid_min_y)

    grid_res = [int(grid_max_x - grid_min_x), int(grid_max_y - grid_min_y)]
    print ('grid_res: ', grid_res)

    visit_counter = np.zeros(grid_res, dtype=np.int32)
    occupied_counter = np.zeros(grid_res, dtype=np.int32)

    print ('grid extends from ({:f}, {:f}) to ({:f}, {:f})'.format(grid_min_x, grid_min_y, grid_max_x, grid_max_y))

    grid_cell_size_x = (grid_max_x - grid_min_x) / float(grid_res[0])
    grid_cell_size_y = (grid_max_y - grid_min_y) / float(grid_res[1])

    norm_factor_x = float(grid_res[0] - 1) / float(grid_max_x - grid_min_x)
    norm_factor_y = float(grid_res[1] - 1) / float(grid_max_y - grid_min_y)
    print ('norm_factor_x: ', norm_factor_x)
    print ('norm_factor_y: ', norm_factor_y)

    for point_id in range(5467):
        point_x = int(_x[0])
        point_y = int(_y[2])

        visit_counter[point_x, point_y] += 1

    free_thresh = 0.55
    occupied_thresh = 0.50
    # draw map
    grid_map = np.zeros(grid_res, dtype=np.float32)
    grid_map_thresh = np.zeros(grid_res, dtype=np.uint8)
    for x in range(grid_res[0]):
        for y in range(grid_res[1]):
            if (visit_counter[x, y] == 0 ):
                grid_map[x, y] = 0.5
            if (visit_counter[x, y] > 0 ):
                print(x, y)

            if grid_map[x, y] >= free_thresh:
                grid_map_thresh[x, y] = 255
            else:
                grid_map_thresh[x, y] = 0


    cv2.imshow("output", grid_map_thresh)
    cv2.waitKey(0)

    # map = np.zeros((100,100),dtype=bool)
    # i = 0; j = 0
    # for i in range(5467):
    #     map[int(_x[i]),int(_y[i])] = True
    #     # map[i,j] = random.randint(a=0,b=1)
    # # print(map[:10,:10])
    # # random.randint(a=0,b=1)
    # plt.xticks(size=12,color = "black")
    # plt.yticks(size=12,color = "black")
    # # plt.figure(figsize=(10,10))
    # plt.imshow(map,cmap="plasma")       # True = yellow
    # plt.show()

    # for i in range(_x.shape[1]):
    #     # print(xyz[0,i],xyz[1,i])
    #     # depth = xyz[2,i]
    #     # plt.scatter(xyz[0,i],xyz[1,i], s=2, c=depth, edgecolors='none', cmap='jet')
    #     # print(_x[0,i],_y[0,i])
    #     plt.plot(_x[0,i],_y[0,i],'.',color='blue')
    #     # plt.pause(1)
    # plt.show()



    cv2.destroyAllWindows()