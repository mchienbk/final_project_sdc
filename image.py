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
###############################################################################

import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np

import my_params


BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_image(image_path, model=None):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    img = demosaic(img, pattern)
    if model:
        img = model.undistort(img)

    return np.array(img).astype(np.uint8)


if __name__ == '__main__':

    # Play undistorce image
    
    import os
    import argparse
    import cv2
    import time
    from datetime import datetime as dt
    from camera_model import CameraModel


    args = argparse.ArgumentParser(description='Preprocess and save all images')
    args.add_argument('--dir', type=str, default=None, help='Directory containing images.')
    args.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
    args.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')

    args.parse_args()

    # Set argument
    # parser = arg_parse()
    args.dir = my_params.dataset_patch + 'stereo\\centre'
    args.models_dir = my_params.project_patch + 'models'
    args.scale = 1.0

    frames = 0
    start = time.time() 

    camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)
    
    timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file")

    model = CameraModel(args.models_dir, args.dir)

    current_chunk = 0
    timestamps_file = open(timestamps_path)
    for line in timestamps_file:
        tokens = line.split()
        datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        chunk = int(tokens[1])

        filename = os.path.join(args.dir, tokens[0] + '.png')
        if not os.path.isfile(filename):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue

        current_chunk = chunk

        img = load_image(filename, model)

        cv2.imshow("img",img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        # cv2.imwrite('data/' + tokens[0] + '.png',img)