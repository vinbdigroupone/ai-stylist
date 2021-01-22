from __future__ import division
import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset

from mmcv import Config
from mmcv.runner import load_checkpoint

# from mmfashion.utils import get_img_tensor

import numpy as np
from tqdm import tqdm

import cv2
import time
import faiss
import pickle


import torchvision.transforms as transforms
from PIL import Image, ImageDraw


def get_img_tensor(img, use_cuda, get_size=False):
    original_w, original_h = img.size

    img_size = (224, 224)  # crop image to (224, 224)
    img.thumbnail(img_size, Image.ANTIALIAS)
    img = img.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    if get_size:
        return img_tensor, original_w, original_w
    else:
        return img_tensor

# Create dataloader and Dataset to load images to get embeddings
class Images(Dataset):
    def __init__(self, data, category, root_dir, gpu=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.gpu = gpu
        self.root_dir = root_dir
        self.data = data
        self.category = category
        self.item_ids = data[category]

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        imgname = self.item_ids[idx] + '.jpg'
        tensor = get_img_tensor(os.path.join(self.root_dir, imgname), self.gpu)
        tensor = torch.squeeze(tensor)
        return tensor

def make_loader(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def create_embedding_spaces(types:list, data, model, gpu=True): # DEMO
    '''
    Embedding_spaces has shape: (n_items, n_type_spaces, n_dims)
      - n_items: number of items
      - n_type_spaces: number of type-embedding spaces i.e. ('top', 'bottoms')
      - n_dims: Number of dimension of an item's embedding in 1 type-embedding space
    '''
    embeddings = dict()
    model.eval()

    for category in types:
        img_tensors = []
        root_dir = 'data/Polyvore/images'
        dataset = Images(data, category, root_dir, gpu=gpu)
        dataloader = make_loader(dataset)
        
        # Convert images to tensor
        for img_tensor in tqdm(dataloader):
            embed = model(img_tensor, return_loss=False)
            img_tensors.append(embed.detach())
        img_tensors = torch.cat(img_tensors) 
        reduced_dim_embed = img_tensors[:, list(type_spaces.values()), :]

        embeddings[category] = reduced_dim_embed.cpu().numpy()
        new_type_spaces = list(type_spaces.keys())

    return embeddings, new_type_spaces

class SearchEngine:
    def __init__(self, n_dims, n_trees, data):
        self.n_dims = n_dims
        self.n_trees = n_trees
        self.data = data

    # 'Index' is similar to a database for query
    def build_index(self, embedding, type_spaces, nlist = 100):
        # List of search indexes for each category(tops, bottoms, shoes)
        self.indexes = dict()

        for category in embedding:
            type_space_index = dict()
            for i, type_space in enumerate(type_spaces):

                nlist = 100
                quantizer = faiss.IndexFlatL2(self.n_dims)  # the other index
                index = faiss.IndexIVFFlat(quantizer, self.n_dims, nlist)
                assert not index.is_trained
                embedding_vec = embedding[category][:, i, :]
                index.train(np.ascontiguousarray(embedding_vec))
                assert index.is_trained
                index.add(np.ascontiguousarray(embedding_vec))

                type_space_index[type_space] = index

            self.indexes[category] = type_space_index

    def search_outfit(self, in_embed, in_type, type_spaces):
        types_to_predict = ''
        if in_type == 'tops':
            types_to_predict = ['bottoms', 'shoes']
        elif in_type == 'shoes':
            types_to_predict = ['bottoms', 'top']
        elif in_type == 'bottoms':
            types_to_predict = ['tops', 'shoes']

        matching_item_ids = list()
        for type_to_predict in types_to_predict:
            # Find index position of the required embedding space in the
            # Embedding matrix
            if (in_type, type_to_predict) in type_spaces:
                type_space_name = (in_type, type_to_predict)
                col_index = type_spaces.index(type_space_name)
            else:
                type_space_name = (type_to_predict, in_type)
                col_index = type_spaces.index(type_space_name)

            index_to_use = self.indexes[type_to_predict][type_space_name]
            similar_img_id = search_matching_item(index_to_use, in_embed[col_index], 
                                                  n_neighbours=1)
            similar_img_id = self.data[type_to_predict][similar_img_id]
            
            matching_item_ids.append(similar_img_id)

            in_type = type_to_predict

        return matching_item_ids

def get_img_embedding(image, model, gpu=True):
    tensor = get_img_tensor(image, gpu)
    model.eval()
    embed = model(tensor, return_loss=False).detach().cpu().numpy()
    embed = np.squeeze(embed)
    return embed

def search_matching_item(index, in_embed, n_neighbours):
    in_embed = np.expand_dims(in_embed, axis=0)
    start = time.time()
    D, I = index.search(in_embed, n_neighbours)
    end = time.time()
    print('Search time:', end - start)

    similar_img_id = np.squeeze(I)
    return similar_img_id

def generate_outfit(image, item_type:str, engine, model, \
                    type_spaces, gpu, data_path='data/Polyvore/images'):
    '''
    Args:
        img_path: path of input image
        image: PIL.Image object of input image
        item_type: type of input item (tops, bottoms, shoes)
        engine: SearchEngine object for search items in outfit
        model: Model used to predict embedding of input image
        
    Returns:
        item_paths: List of paths of items in predicted outfit 
    '''
    in_embed = get_img_embedding(image, model, gpu)
    item_ids = engine.search_outfit(in_embed, item_type, type_spaces)
    
    item_paths = [f'{data_path}/{id_}.jpg' for id_ in item_ids]
    return item_paths