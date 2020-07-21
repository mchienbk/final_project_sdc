import random
import os
import seaborn as sns
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
# from config import *

if __name__ == "__main__":
   
    # Making save
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter('img/video_output2.mp4', fourcc, 30, (600, 800))

    # Load data
    timestamps_path = 'D:/GoogleDrive/Data/20140514/ldmrs.timestamps'
    lidar_folder_path = 'D:/GoogleDrive/Data/20140514/ldmrs/'
    # timestamps_path = 'D:/GoogleDrive/Data/20140514/lms_front.timestamps'
    # lidar_folder_path = 'D:/GoogleDrive/Data/20140514/lms_front/'
    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            timestamp = int(line.split(' ')[0])
            # if (i > 100): break
            print(timestamp)
            
            scan_path = os.path.join(lidar_folder_path + str(timestamp) + '.bin')

            scan_file = open(scan_path)

            scan = np.fromfile(scan_file, np.double)
            scan_file.close()
            scan = scan.reshape((len(scan) // 3, 3)).transpose()

            img = np.zeros((600,800,3),np.uint8)

            # Create a named colour
            red = [255,0,0]

            # Change one pixel
            for i in range(scan.shape[1]):
                x = 5*scan[0][i]
                y = 5*scan[1][i]
                if (x > 599) : x = 599
                if (x < 1) : x = 1
                if (y < -400) : y = -399 
                if (y > 400) : y = 399 
                img[600-(int(x)),(int(y)) + 400]=red


            # Save
            cv2.imshow("result",img)
            vw.write(img)

            # Press 'q' to exit!
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    print('done!')
    cv2.destroyAllWindows()