import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint
from visualization import board_add_image, board_add_images, save_images
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import json
import networks
import matplotlib.pyplot as plt
from inference import tryon

image_path = './data/train/image/000003_0.jpg'
image_parse_path = './data/train/image-parse/000003_0.png'
cloth_path = './data/train/cloth/000003_1.jpg'
cloth_mask_path = './data/train/cloth-mask/000003_1.jpg'
pose_path = './data/train/pose/000003_0_keypoints.json'
tom_model_path = './model/tom_step_200000.pth'
gmm_model_path = './model/gmm_step_200000.pth'
grid_path = 'grid.png'

image = Image.open(image_path)
image_parse = Image.open(image_parse_path)
cloth = Image.open(cloth_path)
cloth_mask = Image.open(cloth_mask_path)
with open(pose_path, 'r') as f: 
    pose = json.load(f)

output = tryon(image, image_parse, cloth, cloth_mask, pose)

plt.imshow(output)
plt.show()