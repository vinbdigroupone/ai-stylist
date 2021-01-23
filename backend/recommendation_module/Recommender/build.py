from __future__ import division
import os
import time
import pickle
import torch

from mmcv import Config

from .engine import generate_outfit
from .engine import SearchEngine


def build_system():
    '''
    Prepare model and database for query
    '''
    ############ SYSTEM SETUP ############
    start_time = time.time()

    cfg = Config.fromfile('./recommendation_module/Recommender/configs/fashion_recommendation/type_aware_recommendation_polyvore_disjoint_l2_embed.py')
    cfg.load_from = './recommendation_module/Recommender/checkpoint/epoch_16.pth'

    # init distributed env first
    distributed = False
    gpu = torch.cuda.is_available()
    if gpu:
        model_path = './recommendation_module/Recommender/save_files/model_gpu.pickle'
    else:
        model_path = './recommendation_module/Recommender/save_files/model_cpu.pickle'

    types = ['tops', 'bottoms', 'shoes']
    type_spaces = {('tops', 'bottoms'): 9,
                    ('shoes', 'tops'): 5,
                    ('shoes', 'bottoms'): 6}

    # Load model and data from pickle file
    with open('./recommendation_module/Recommender/save_files/data_category.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('./recommendation_module/Recommender/save_files/model_cpu.pickle', 'rb') as f:
        model = pickle.load(f)
    with open('./recommendation_module/Recommender/save_files/embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)
    with open('./recommendation_module/Recommender/save_files/new_type_spaces.pickle', 'rb') as f:
        new_type_spaces = pickle.load(f)

    # embeddings, new_type_spaces = create_embedding_spaces(types, data, model)  

    engine = SearchEngine(256, 10, data)
    engine.build_index(embeddings, type_spaces)

    end_time = time.time()
    print('BUILD TIME: ', end_time-start_time)
    ######################################

    return engine, model, new_type_spaces, gpu

if __name__ == '__main__':
    engine, model, new_type_spaces, gpu = build_system()

    in_img_path = './recommendation_module/Recommender/shirt.png'
    item_type = 'tops'
    item_paths = generate_outfit(in_img_path, item_type, \
                                engine, model, new_type_spaces, gpu=gpu)


    # import matplotlib.pyplot as plt
    # import cv2
    # for path in item_paths:
    #     img = cv2.imread(path)
    #     plt.imshow(img)
    #     plt.show()