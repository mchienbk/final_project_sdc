

import os
import cv2
import numpy as np
import pandas as pd
import time

from datetime import datetime as dt
import matplotlib.pyplot as plt

import my_params

# import random
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# from math import *


def play_lidar():
   
    # Making save
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (600, 800))

    # Load data
    timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'
    lidar_folder_path = my_params.laser_dir

    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            # if (i > 100): break

            timestamp = int(line.split(' ')[0])

            datetime = dt.utcfromtimestamp(timestamp/1000000)
            print(datetime)

            file_path = os.path.join(lidar_folder_path + '\\' + str(timestamp) + '.bin')
            scan_file = open(file_path)

            data = np.fromfile(scan_file, np.double)
            scan_file.close()
            data = data.reshape((len(data) // 3, 3)).transpose()

            # Create a color
            color = [255,0,0]

            # Blank image
            img = np.zeros((600,800,3),np.uint8)

            # Change one by one pixel
            for i in range(data.shape[1]):
                x = 5*data[0][i]
                y = 5*data[1][i]
                if (x > 599) : x = 599
                if (x < 1) : x = 1
                if (y < -400) : y = -399 
                if (y > 400) : y = 399 
                img[600-(int(x)),(int(y)) + 400]=color

            # Save
            cv2.imshow("result",img)
            vw.write(img)

            # Press 'q' to exit!
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    print('done!')
    cv2.destroyAllWindows()


def play_map():
    # Round truth
    rtk_directory = my_params.project_patch + 'groundtruth\\groundtruth.csv'

    rtk = pd.read_csv(rtk_directory) 
    rtk.head()

    # Get size
    BBox = (rtk.longitude.min(), rtk.longitude.max(), 
            rtk.latitude.min(),  rtk.latitude.max())

    print(BBox)

    # image from opestreetmap.org
    ruh_m = plt.imread(my_params.project_patch + 'groundtruth\\groundtruth.png')

    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(rtk.longitude, rtk.latitude, zorder=1, alpha= 0.2, c='b', s=10)
    ax.set_title('Plotting Data on Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

    plt.show()

def play_gps():
    # Read data 
    input_INS = my_params.dataset_patch + 'gps//ins.csv'
    input_GPS = my_params.dataset_patch + 'gps//gps.csv'

    ins = pd.read_csv(input_INS) 
    ins.head()

    gps = pd.read_csv(input_GPS) 
    gps.head()

    # Get size
    BBox = (gps.longitude.min(), gps.longitude.max(), 
            gps.latitude.min(),  gps.latitude.max())

    # image from opestreetmap.org
    ruh_m = plt.imread(my_params.project_patch + 'groundtruth\\groundtruth.png')

    fig, ax = plt.subplots(figsize = (8,7))

    # Plot INS 'red'
    ax.scatter(ins.longitude, ins.latitude, zorder=1, alpha= 0.2, c='r', s=10)


    # Plot GPS 'blue'
    ax.scatter(gps.longitude, gps.latitude, zorder=1, alpha= 0.2, c='b', s=10)

    ax.set_title('Plotting GPS/INS on Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

    plt.show()

def play_processed_img():

    # Get image    
    image_dir = my_params.image_dir
    model_dir = my_params.model_dir
    reprocess_dir = my_params.reprocess_image_dir
    camera_name = my_params.camera_name
    scale = 0.1

    frames = 0
    start = time.time() 
    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera_name + '.timestamps'))

    current_chunk = 0
    timestamps_file = open(timestamps_path)
    for line in timestamps_file:
        tokens = line.split()
        datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        chunk = int(tokens[1])

        filename = os.path.join(reprocess_dir + '//' + tokens[0] + '.png')

        if not os.path.isfile(filename):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue

        current_chunk = chunk
        
        # read and resize image
        img = cv2.imread(filename)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow("img",img)
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

        frames += 1
        print(datetime)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

def play_vo():
    import numpy.matlib as ml
    import csv
    from interpolate_poses import interpolate_poses
    from transform import build_se3_transform

    # Read data
    vo_directory = my_params.project_patch + 'groundtruth\\vo.csv'

    H_new=np.identity(4)
    C_origin=np.zeros((3,1))
    R_origin=np.identity(3)
    abs_poses = [ml.identity(4)]

    x_old=(0,0)
    z_old=(0,0)

    fig = plt.figure()
    gs = plt.GridSpec(2,3)

    with open(vo_directory) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        for row in vo_reader:
            timestamp = int(row[0])
            datetime = dt.utcfromtimestamp(timestamp/1000000)
            print(datetime)

            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            # print('rel pose',rel_pose)

            # abs_pose = abs_poses[-1] * rel_pose
            # abs_poses.append(abs_pose)

            H_new=H_new@rel_pose
            x_test=H_new[0,3]
            z_test=H_new[2,3]

            # print("old: ",(x_old,z_old))
            # print("new: ",(x_test,z_test))
            print((x_old,z_old), " : ",(x_test,z_test))

            x_old=x_test
            z_old=z_test

            plt.plot(x_test,-z_test,'o',color='blue')
            plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    print("Run OK")

    play_processed_img()