import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from .cp_dataset import CPDataset, CPDataLoader
from .networks import GMM, UnetGenerator, load_checkpoint
from .visualization import board_add_image, board_add_images, save_images
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import json
import matplotlib.pyplot as plt 

GRID_PATH = './tryon_module/grid.png'
TOM_MODEL_PATH = './tryon_module/model/tom_step_200000.pth'
GMM_MODEL_PATH = './tryon_module/model/gmm_step_200000.pth'
FINE_HEIGHT = 256
FINE_WIDTH = 192
RADIUS = 5
GRID_SIZE = 5


#image_parse -> Image object
#image -> Image object 
#cloth -> Image object
#cloth_mask -> Image object 
#pose is json loaded as f pose = json.load(pose_file)
def tryon(image, image_parse, cloth, cloth_mask, pose):
  transform_im = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  parse_array = np.array(image_parse)
  parse_shape = (parse_array > 0).astype(np.float32)
  parse_head = (parse_array == 1).astype(np.float32) + \
              (parse_array == 2).astype(np.float32) + \
              (parse_array == 4).astype(np.float32) + \
              (parse_array == 13).astype(np.float32)
  phead = torch.from_numpy(parse_head)
  im = transform_im(image)
  im_h = im * phead - (1 - phead)
  #pose is json loaded 
  parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
  parse_shape = parse_shape.resize((FINE_WIDTH // 16, FINE_HEIGHT // 16), Image.BILINEAR)
  parse_shape = parse_shape.resize((FINE_WIDTH, FINE_HEIGHT), Image.BILINEAR)

  shape = transform_im(parse_shape) 
  pose_data = pose['people'][0]['pose_keypoints']
  pose_data = np.array(pose_data)
  pose_data = pose_data.reshape((-1, 3))

  point_num = pose_data.shape[0]
  pose_map = torch.zeros(point_num, FINE_HEIGHT, FINE_WIDTH)
  r = RADIUS
  im_pose = Image.new('L', (FINE_HEIGHT, FINE_WIDTH))
  pose_draw = ImageDraw.Draw(im_pose)
  for i in range(point_num):
      one_map = Image.new('L', (FINE_WIDTH, FINE_HEIGHT))
      draw = ImageDraw.Draw(one_map)
      pointx = pose_data[i, 0]
      pointy = pose_data[i, 1]
      if pointx > 1 and pointy > 1:
          draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
          pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
      one_map = transform_im(one_map)
      pose_map[i] = one_map[0]
  agnostic = torch.cat([shape, im_h, pose_map])  # (22, 256, 192)
  agnostic = torch.unsqueeze(agnostic, 0)
  c = transform_im(cloth)  # (3, 256, 192)
  cm_array = np.array(cloth_mask)
  cm_array = (cm_array >= 128).astype(np.float32)
  cm = torch.from_numpy(cm_array)
  cm.unsqueeze_(0)
  c = torch.unsqueeze(c, 0)
  cm = torch.unsqueeze(cm, 0)
  im_g = Image.open(GRID_PATH)
  im_g = transform_im(im_g)  # (3, 256, 192)
  im_g = torch.unsqueeze(im_g, 0)
  gmm = GMM(FINE_HEIGHT, FINE_WIDTH, GRID_SIZE)
  gmm.load_state_dict(torch.load(GMM_MODEL_PATH))
  with torch.no_grad():
    grid, theta = gmm(agnostic, c)  # grid (256, 192, 2), theta (18)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')  # (2, 356, 192)
    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')  # (1, 256, 192)
    warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros') 
  tom = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
  tom.load_state_dict(torch.load(TOM_MODEL_PATH))
  with torch.no_grad():
    outputs = tom(torch.cat([agnostic, c], 1))
    p_rendered, m_composite = torch.split(outputs, 3, 1)
    p_rendered = F.tanh(p_rendered)
    m_composite = F.sigmoid(m_composite)
    p_tryon = c * m_composite + p_rendered * (1 - m_composite)  # (3, 256, 192)
  tensor = (p_tryon.clone()+1)*0.5*255
  tensor = tensor.cpu().clamp(0,255)
  output = tensor[0].permute(1, 2, 0)
  output = output.numpy().astype('uint8')
  return output