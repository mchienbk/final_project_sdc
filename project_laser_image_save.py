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

image_dir = my_params.image_dir
laser_dir = my_params.laser_dir
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

# print('index timestamp:', timestamp)
# print('start time:', start_time)
# print('end time:', end_time)
        timestamp = int(line.split(' ')[0])
        start_time = timestamp - 2e6
        # if start_time < 1446206199046687:
        #     start_time = 1446206199046687
        
        if i < 4: 
            image_path = os.path.join(image_dir, str(timestamp) + '.png')
            image = load_image(image_path, model)
            plt.imshow(image)
            plt.xlim(0, image.shape[1])
            plt.ylim(image.shape[0], 0)
            plt.xticks([])
            plt.yticks([])

            plt.savefig(output_dir + '\\yolo_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png')
            continue
        pointcloud = []
        pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,
                                           start_time, timestamp + 2e6, timestamp)

        pointcloud = np.dot(G_camera_posesource, pointcloud)

        image_path = os.path.join(image_dir, str(timestamp) + '.png')
        image = load_image(image_path, model)

        uv, depth = model.project(pointcloud, image.shape)


        plt.figure(1)
        plt.imshow(image)
        plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(output_dir + '\\yolo_img_' + my_params.dataset_no + '\\' + str(timestamp) + '.png')
        print("save-",i)
# plt.show()



