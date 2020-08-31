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
import cv2
from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

import my_params

image_dir = my_params.image_dir
# laser_dir = my_params.laser_dir
laser_dir = my_params.lmsfront_dir
poses_file = my_params.poses_file
models_dir = my_params.model_dir
extrinsics_dir = my_params.extrinsics_dir
image_idx = 1572

output_dir = my_params.output_dir

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
        # if i == image_idx:
            # print('index:', i)
            # timestamp = int(line.split(' ')[0])
            # break

        timestamp = int(line.split(' ')[0])
        start_time = timestamp - 2e6
        
        frame_path = os.path.join(my_params.reprocess_image_dir + '//' + str(timestamp) + '.png')
        frame = cv2.imread(frame_path)   # must processed imagte
        # if start_time < 1446206199046687:
        #     start_time = 1446206199046687
        
        if i < 4: 
            # image_path = os.path.join(image_dir, str(timestamp) + '.png')
            # image = load_image(image_path, model)
            # plt.imshow(image)
            # plt.xlim(0, image.shape[1])
            # plt.ylim(image.shape[0], 0)
            # plt.xticks([])
            # plt.yticks([])

            # plt.savefig(output_dir + '\\pointcloud_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png')
            # savefile_name = output_dir + '\\pointcloud_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png'
            savefile_name = output_dir + '\\lms_front_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png'
            cv2.imwrite(savefile_name, frame)           
            continue

        pointcloud = []
        pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,
                                           start_time, timestamp + 2e6, timestamp)

        pointcloud = np.dot(G_camera_posesource, pointcloud)

        # image_path = os.path.join(image_dir, str(timestamp) + '.png')
        # image = load_image(image_path, model)
        # uv, depth = model.project(pointcloud, image.shape)
        uv, depth = model.project(pointcloud, (1280,960,3))

        for k in range(uv.shape[1]):
            x_p = (int)(np.ravel(uv[:,k])[0])
            y_p = (int)(np.ravel(uv[:,k])[1])

            color = (int(255-8*depth[k]),255-3*depth[k],50+3*depth[k])
            pointcloud= cv2.circle(pointcloud, (x_p, y_p), 1, color, 1)
            frame= cv2.circle(frame, (x_p, y_p), 1, color, 1)
            # savefile_name = output_dir + '\\pointcloud_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png'
            savefile_name = output_dir + '\\lms_front_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png'
            cv2.imwrite(savefile_name, frame)

        # plt.figure()
        # plt.imshow(image)
        # plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
        # plt.xlim(0, image.shape[1])
        # plt.ylim(image.shape[0], 0)
        # plt.xticks([])
        # plt.yticks([])

        # plt.savefig(output_dir + '\\pointcloud_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png')
        print("save-",i)

# plt.show()



