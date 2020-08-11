

import os
import cv2
import numpy as np
import pandas as pd

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


def play_vo():
    # Read data
    vo_directory = my_params.project_patch + 'groundtruth\\vo.csv'

    vo = pd.read_csv(vo_directory) 
    vo.head()

if __name__ == "__main__":
    print("Run OK")

    play_vo()