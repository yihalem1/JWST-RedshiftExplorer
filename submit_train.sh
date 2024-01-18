#!/bin/sh

# >>> Job name <<< #
#SBATCH -J 1_pseudo-sr_weak12x12_splited_0814

# >>> Partition name (queue) <<< #
#SBATCH -p pleiades

# >>> Per node <<< #
#SBATCH --nodes=1

# >>> Core per node <<< #
#SBATCH --ntasks-per-node=1

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:1

# >>> Output <<< #
#SBATCH -o %x.%j.o

# >>> Error <<< #
#SBATCH -e %x.%j.e


# python train.py configs/faces.yaml --port 12122
# python train_NTIRE.py configs/NTIRE.yaml --port 12122
# python train_weak_lensing.py configs/WeakLensing.yaml --port 12120
# CUDA_VISIBLE_DEVICES=7 python train_galaxyZoo.py configs/GalaxyZoo.yaml --port 12120
# python train_galaxyZoo.py configs/GalaxyZoo.yaml --port 12120
# python train_galaxyZoo_model.py configs/galaxy.yaml --port 12120
# python train_galaxyZoo_model_unpaired.py configs/galaxy_unparied.yaml --port 12121
#./exec
python train_galaxyZoo_model_unpaired_splited.py configs/galaxy_unpaired_12x12_splited.yaml --port 12121
