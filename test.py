import os
import argparse
from yacs.config import CfgNode

import torch
from models.pseudo_model import Pseudo_Model
from models.galaxy_model import Galaxy_Model
from tools.pseudo_GalaxyZoo_data import GalaxyZoo_test
from tools.utils import save_tensor_image

from datetime import datetime

import numpy as np

from tools.pseudo_face_data import faces_data
import random
import argparse

device = 'cpu'

def parse_option():
    parser = argparse.ArgumentParser()

    # easy config modification
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--test_source_path', default=None, type=str)
    parser.add_argument('--img_save_folder', type=str)
    parser.add_argument('--yaml', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--gt_lr_path', type=str)

    args = parser.parse_args()

    with open(args.yaml, "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()

    return args, CFG
    

def main(args, CFG):
    testset = GalaxyZoo_test(data_lr=args.test_data_path, data_hr=args.test_source_path, gt_lr=args.gt_lr_path, img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)

    print(len(testset))

    for idx in range(len(testset)):
        lr = testset[idx]["lr"].unsqueeze(0).to(device, dtype=torch.float)
        hr = testset[idx]["hr"].unsqueeze(0).to(device, dtype=torch.float) 
        gt_lr = testset[idx]["gt_lr"].unsqueeze(0).to(device, dtype=torch.float)

        save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
        save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_hr.png"), hr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
        save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_gt_lr.png"), gt_lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)

if __name__ == "__main__":
    args, CFG = parse_option()
    
    os.makedirs(args.img_save_folder, exist_ok=True)

    random_seed = CFG.EXP.SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    main(args, CFG)