#!/bin/sh

# >>> Job name <<< #
#SBATCH -J 1_KSU_pse_test

# >>> Partition name (queue) <<< #
#SBATCH -p pleiades

# >>> Per node <<< #
#SBATCH --nodes=1

# >>> Core per node <<< #
#SBATCH --ntasks-per-node=1

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:0

# >>> Output <<< #
#SBATCH -o %x.%j.o

# >>> Error <<< #
#SBATCH -e %x.%j.e


# (1) Train: GZ | Test: GZ
# python test.py \
#     --model_path "/home/s20225004/pseudo-sr/results/GalaxyZoo_Corrupted2/nets/nets_174915.pth" \
#     --test_data_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Noised_12X12" \
#     --test_source_path "/home/s20225004/pseudo-sr/GalaxyZoo/48X48/val_Resized_48X48" \
#     --gt_lr_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Resized_12X12" \
#     --img_save_folder "./results/0817/train_GZ-Test_GZ/" \
#     --yaml "/home/s20225004/pseudo-sr/configs/GalaxyZoo.yaml" \
#     --model_type pseudo
# echo "***1 is done***"

# (2) Train: GZ | Test: WL(resized)
# python test_galaxyZoo.py \
#     --model_path "/home/s20225004/pseudo-sr/results/GalaxyZoo_Corrupted2/nets/nets_174915.pth" \
#     --test_data_path "/home/s20225004/pseudo-sr/resized_weak_lens" \
#     --img_save_folder "./results/0817/train_GZ-Test_RWL/" \
#     --yaml "/home/s20225004/pseudo-sr/configs/GalaxyZoo.yaml" \
#     --model_type pseudo

# echo "***2 is done***"

# (3) Train: GZ + WL(12x12 Only) | Test: GZ
# python test.py \
#     --model_path "/home/s20225004/pseudo-sr/results/weak_zoo_corrupted-patch3_/nets/nets_26781.pth" \
#     --test_data_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Noised_12X12" \
#     --test_source_path "/home/s20225004/pseudo-sr/GalaxyZoo/48X48/val_Resized_48X48" \
#     --gt_lr_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Resized_12X12" \
#     --img_save_folder "./results/0817/train_GZ_WL-Test_GZ/" \
#     --yaml "/home/s20225004/pseudo-sr/configs/galaxy_unpaired.yaml" \
#     --model_type galaxy

# echo "***3 is done***"

# (4) Train: GZ + WL(12x12 Only) | Test: WL(Resized)
# python test_galaxyZoo.py \
#     --model_path "/home/s20225004/pseudo-sr/results/weak_zoo_corrupted-patch3_/nets/nets_26781.pth" \
#     --test_data_path "/home/s20225004/pseudo-sr/resized_weak_lens" \
#     --img_save_folder "./results/0817/train_GZ_WL-Test_RWL/" \
#     --yaml "/home/s20225004/pseudo-sr/configs/galaxy_unpaired.yaml" \
#     --model_type galaxy

# echo "***4 is done***"


# (5) Train: WL(12x12 trainset) | Test: WL(12x12 testset)
python test_galaxyZoo.py \
    --model_path "/home/s20225004/pseudo-sr/results/unpaired_12x12_splited/nets/nets_28045.pth" \
    --test_data_path "/home/s20225004/pseudo-sr/weak_lensing/20220814/weak_12x12_test" \
    --img_save_folder "./results/0822/train_WL-Test_WL/" \
    --yaml "./configs/galaxy_unpaired_12x12_splited.yaml" \
    --model_type galaxy

# (6) Train: WL(12x12 trainset) | Test: GZ
python test_galaxyZoo.py \
    --model_path "/home/s20225004/pseudo-sr/results/unpaired_12x12_splited/nets/nets_28045.pth" \
    --test_data_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Noised_12X12" \
    --test_source_path "/home/s20225004/pseudo-sr/GalaxyZoo/48X48/val_Resized_48X48" \
    --gt_lr_path "/home/s20225004/pseudo-sr/GalaxyZoo/12X12/val_Resized_12X12" \
    --img_save_folder "./results/0822/train_WL-Test_GZ/" \
    --yaml "./configs/galaxy_unpaired_12x12_splited.yaml" \
    --model_type galaxy