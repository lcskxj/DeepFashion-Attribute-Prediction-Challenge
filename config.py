# -*- coding: utf-8 -*-

CATEGORIES = [7, 3, 3, 4, 6, 3]
CROP_SIZE = 224

DATASET_BASE = r'./FashionDataset/'
DUMPED_MODEL = "model_1000_final.pth.tar"



FREEZE_PARAM = True

GPU_ID = 0
NUM_WORKERS = 4
TEST_BATCH_SIZE = 1
IMG_SIZE = 256


EPOCH = 20
INTER_DIM = 512
LR = 1e-4
TRAIN_BATCH_SIZE = 64


# 8, 1024, 1e-4, 64: 80.6%