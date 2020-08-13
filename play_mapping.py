


import os
import cv2
import csv
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy.matlib as ml

import my_params

from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from transform import build_se3_transform
from camera_model import CameraModel

# Read data 
input_INS = my_params.dataset_patch + 'gps//ins.csv'
input_GPS = my_params.dataset_patch + 'gps//gps.csv'

H_new=np.identity(4)
abs_poses = [ml.identity(4)]
ins_timestamps = [0]
use_rtk=False

x_old=(0,0)
z_old=(0,0)
fig = plt.figure()
gs = plt.GridSpec(2,3)

startpoint = [620163.607591,5736385.159098]

with open(input_INS) as ins_file:
    ins_reader = csv.reader(ins_file)
    headers = next(ins_file)

    index = 0
    for row in ins_reader:
        timestamp = int(row[0])
        ins_timestamps.append(timestamp)

        x = float(row[6]) - float(startpoint[0])
        y = float(row[5]) - float(startpoint[1])
        if (index<1000):
            plt.plot(x,y,'.',color='red')
        else:
            plt.plot(x,y,'.',color='blue')
        index = index + 1

        if (index>5000):
            break

plt.show()

cv2.waitKey(0)
ins_timestamps = ins_timestamps[1:]
abs_poses = abs_poses[1:]