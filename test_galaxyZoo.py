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

device = 'cuda'

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
    print(args)
    testset = GalaxyZoo_test(data_lr=args.test_data_path, data_hr=args.test_source_path, gt_lr=args.gt_lr_path, img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)

    if args.model_type == "galaxy":
        model = Galaxy_Model(device, CFG)
    elif args.model_type == "pseudo":
        model = Pseudo_Model(device, CFG)

    model.net_load(args.model_path)
    model.mode_selector("eval")

    print(len(testset))

    results_list = []

    for b in range(len(testset)):
        if b % 1000 == 0:
            print(b)

        lr = testset[b]["lr"].unsqueeze(0).to(device, dtype=torch.float)
        
        y, sr, _ = model.test_sample(lr)
        
        if args.test_source_path is not None:
            hr = testset[b]["hr"].unsqueeze(0).to(device, dtype=torch.float) 
            gt_lr = testset[b]["gt_lr"].unsqueeze(0).to(device, dtype=torch.float)
            results_list.append((lr, y, sr, hr, gt_lr))
        else:
            results_list.append((lr, y, sr))

    print("[saving Start] total: ", len(results_list))

    if args.test_source_path is not None:
        for idx, (lr, y, sr, hr, gt_lr) in enumerate(results_list):
            if idx % 1000 == 0:
                print("saving counts: ", idx)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_sr.png"), sr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_hr.png"), hr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_gt_lr.png"), gt_lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)

    else:
        for idx, (lr, y, sr) in enumerate(results_list):
            if idx % 1000 == 0:
                print("saving counts: ", idx)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_sr.png"), sr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)
            save_tensor_image(os.path.join(args.img_save_folder, f"{idx:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB, CFG.DATA.CHANNEL)

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