'''
COTR demo for a single image pair
'''
import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)

def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread(opt.img_a, pilmode='RGB')
    img_b = imageio.imread(opt.img_b, pilmode='RGB')

    engine = SparseEngine(model, 32, mode='tile')
    t0 = time.time()

    opt.queries_a = None
    if opt.query_wnd is not None:
        left, top, right, bottom = opt.query_wnd.split(',')
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        m = bottom - top + 1
        n = right - left + 1
        opt.queries_a = np.zeros((m*n,2))
        for i in range(m):
            for j in range(n):
                opt.queries_a[i*n+j][0] = j+left
                opt.queries_a[i*n+j][1] = i+top
        opt.max_corrs = opt.queries_a.shape[0]

    corrs = engine.cotr_corr_multiscale_with_cycle_consistency(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=opt.max_corrs, queries_a=opt.queries_a)
    t1 = time.time()

    opt.correspondences_path = os.path.join(opt.out_dir, opt.correspondences)
    with open(opt.correspondences_path, 'wb') as f:
        np.save(f, corrs)

    utils.visualize_corrs(img_a, img_b, corrs)
    print(f'spent {t1-t0} seconds for {opt.max_corrs} correspondences.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--img_a', type=str, default='./sample_data/imgs/cathedral_1.jpg', help='img a path')
    parser.add_argument('--img_b', type=str, default='./sample_data/imgs/cathedral_2.jpg', help='img b path')
    parser.add_argument('--correspondences', type=str, default='correspondences.npy', help='file name to output correspondences')
    parser.add_argument('--query_wnd', type=str, default=None, help='query window: left,top,right,bottom coordinates')
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--max_corrs', type=int, default=100, help='number of correspondences')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    print_opt(opt)

    main(opt)
