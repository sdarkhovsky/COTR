import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

import open3d as o3d

from COTR.utils import utils, debug_utils
from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)

def visualize_points(X):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    m = 'o'
    ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel('Camera X')
    ax.set_ylabel('Camera Y')
    ax.set_zlabel('Camera Z')

    plt.show()

    return


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
    # 59.9696 is probably baseline in pixels, 0.05997 is the baseline in meters.
    # In our derivations the baseline is used in C in the transformation RX+C between
    # camera cooridnate systems, where X and C are in meters. So we use the value in meters.
    baseline=0.05997
    # visually found correspondence: (231,52) im0 <-> (221,52) im1
    # corrs = np.array([[231,52,221,52]])

    points = utils.points_from_correspondence(corrs, fx, fy, cx, cy, baseline).transpose()
    visualize_points(points)

    img_a = imageio.imread(opt.img_a, pilmode='RGB')
    img_b = imageio.imread(opt.img_b, pilmode='RGB')

    colors = (img_a[tuple(np.floor(corrs[:, :2]).astype(int)[:, ::-1].T)] / 255 + img_b[tuple(np.floor(corrs[:, 2:]).astype(int)[:, ::-1].T)] / 255) / 2
    colors = np.array(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--correspondences', type=str, default='correspondences.npy', help='file name to output correspondences')
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--img_a', type=str, default='./sample_data/imgs/storage_room_2_2l_im0.png', help='img a path')
    parser.add_argument('--img_b', type=str, default='./sample_data/imgs/storage_room_2_2l_im1.png', help='img b path')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    print_opt(opt)

    main(opt)
