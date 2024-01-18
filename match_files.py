from glob import glob
import numpy as np
import cv2
import os
import shutil

gt_48_path = "/home/s20225004/pseudo-sr/GalaxyZoo/48X48/val_Resized_48X48"
gt_12_path = "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Resized_12X12"

hr_path = "/home/s20225004/pseudo-sr/results/weak_zoo_corrupted-patch3_nets_26781_train_upaired_test_galaxyzoo/hr"
lr_path = "/home/s20225004/pseudo-sr/results/weak_zoo_corrupted-patch3_nets_26781_train_upaired_test_galaxyzoo/lr"


gt_48_files = glob(os.path.join(gt_48_path, "**/*.png"), recursive=True)
gt_12_files = glob(os.path.join(gt_12_path, "**/*.png"), recursive=True)

hr_files = glob(os.path.join(hr_path, "**/*.png"), recursive=True)
lr_files = glob(os.path.join(lr_path, "**/*.png"), recursive=True)

assert len(hr_files) == len(gt_48_files)
assert len(lr_files) == len(gt_12_files)

for gt_48_file in gt_48_files:  
    gt_48_img = cv2.imread(gt_48_file, cv2.IMREAD_GRAYSCALE)
    for hr_file in hr_files:
        hr_img = cv2.imread(hr_file, cv2.IMREAD_GRAYSCALE)

        hh = gt_48_img - hr_img
        
        if hh.sum() == 0:
            print("Gt: ",gt_48_file)
            print("Hr: ",hr_file)
            print("Diff: ", hh.sum())
            shutil.copy2(gt_48_file, os.path.join("/home/s20225004/pseudo-sr/matching", hr_file.split("/")[-1].split("_")[0] + "_gt.png"))
            break
exit()

for gt_48_file, hr_file in zip(gt_48_files, hr_files):  
    gt_48_img = cv2.imread(gt_48_file, cv2.IMREAD_GRAYSCALE)
    hr_img = cv2.imread(hr_file, cv2.IMREAD_GRAYSCALE)

    hh = gt_48_img - hr_img
    
    if hh.sum() != 0:
        print("Gt: ",gt_48_file)
        print("Hr: ",hr_file)
        print("Diff: ", hh.sum())
        print("Not the Same at 48x48")
        break

for gt_12_file, lr_file in zip(gt_12_files, lr_files):  
    gt_12_img = cv2.imread(gt_12_file, cv2.IMREAD_GRAYSCALE)
    lr_img = cv2.imread(lr_file, cv2.IMREAD_GRAYSCALE)

    hh = gt_12_img - lr_img
    
    if hh.sum() != 0:
        print("Not the Same at 12x12")
        break