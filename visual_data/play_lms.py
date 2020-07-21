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
    vw = cv2.VideoWriter('img/lms.mp4', fourcc, 30, (600, 800))

    # Load data
    timestamps_path = 'D:/GoogleDrive/Data/20140514/lms_rear.timestamps'
    lidar_folder_path = 'D:/GoogleDrive/Data/20140514/lms_rear/'
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
            red = [0,255,0]

            # Change one pixel
            for i in range(scan.shape[1]):
                x = 20*scan[0][i]
                y = 20*scan[1][i]
                # if (x > 599) : x = 599
                # if (x < 1) : x = 1
                # if (y < -400) : y = -399 
                # if (y > 400) : y = 399 
                x = np.min([(-x+300),599])
                y = np.min([(y + 400),799])
                img[int(x),int(y)]=red


            # Save
            cv2.imshow("result",img)
            vw.write(img)

            # Press 'q' to exit!
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
    print('done!')
    cv2.destroyAllWindows()