import argparse
import os
import time

import cv2
import numpy as np
import torch
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
    with open(opt.correspondences_path, 'rb') as f:
        corrs = np.load(f)
    utils.visualize_depth(opt, corrs=corrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--correspondences', type=str, default='correspondences.npy', help='file name to output correspondences')
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    print_opt(opt)

    main(opt)
