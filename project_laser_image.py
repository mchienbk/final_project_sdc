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

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, default=my_params.image_dir, help='Directory containing images')
parser.add_argument('--laser_dir', type=str, default=my_params.laser_dir, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, default=my_params.dataset_patch + 'gps//ins.csv' ,help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, default=my_params.model_dir, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, default=my_params.extrinsics_dir, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

# Try to add argument manual #
args.image_idx = 1570


model = CameraModel(args.models_dir, args.image_dir)

extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

print("G_camera_vehicle",G_camera_vehicle)

poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
if poses_type in ['ins', 'rtk']:
    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

timestamps_path = os.path.join(my_params.dataset_patch + model.camera + '.timestamps')

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if (i+1) == args.image_idx:
            print('index:', i)
            start_time = int(line.split(' ')[0])
        if i == args.image_idx:
            print('index:', i)
            timestamp = int(line.split(' ')[0])
        if (i-1) == args.image_idx:
            print('index:', i)
            end_time = int(line.split(' ')[0])
            break

print('index timestamp:', timestamp)
print('start time:', start_time)
print('end time:', end_time)

pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
                                           timestamp - 1e6, timestamp + 1e6, timestamp)

pointcloud = np.dot(G_camera_posesource, pointcloud)

image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
image = load_image(image_path, model)

# Print filename
print('image patch: ',image_path)

uv, depth = model.project(pointcloud, image.shape)

plt.imshow(image)
plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
plt.xlim(0, image.shape[1])
plt.ylim(image.shape[0], 0)
plt.xticks([])
plt.yticks([])
plt.show()
