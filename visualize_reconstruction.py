import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from COTR.utils import utils, debug_utils
from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)

def main(opt):
    opt.correspondences_path = os.path.join(opt.out_dir, opt.correspondences)
    corrs = np.load(opt.correspondences_path)

    # see calib.txt,im0.png,im1.png,images.txt
    #   in olgplex/datasets/ETH3D/two_view_test/storage_room_2_2l
    # from calib.txt
    fx = 545.506   # distance from the camera center to the image plane expressed in pixels
    fy = 545.506
    cx = 442.452   # pixels
    cy = 241.95
    #
    # distance between camera centers:
    # baseline=59.9696 in calib.txt
    # but difference between TX for img0 and img1 is 4.63527-4.5753=0.05997 in images.txt
    # which one should be used?
    baseline=0.05997
    # visually found correspondence: (231,52) im0 <-> (221,52) im1
    # corrs = np.array([[231,52,221,52]])

    utils.visualize_depth(corrs, fx, fy, cx, cy, baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--correspondences', type=str, default='correspondences.npy', help='file name to output correspondences')
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    print_opt(opt)

    main(opt)
