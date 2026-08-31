## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
from glob import glob
from corun_colabator.archs.corun_arch import CORUN
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Dehazing using CORUN')

parser.add_argument('--input_dir', default='/home/nfs/fcy/Datasets/Dehaze/RTTS/JPEGImages', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_weights/CORUN.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='RTTS', type=str, help='Test Dataset')
parser.add_argument('--opt', default='../options/valid_corun.yml', type=str, help='options')
parser.add_argument('--size', default=0, type=int,
                    help='Resize each input to SIZE x SIZE before inference. '
                         'Default 0 keeps the native resolution, which is what '
                         'the reported results use. Set e.g. --size 256 only if '
                         'you need a fixed small input to fit in memory; note '
                         'that this changes the output resolution and therefore '
                         'the metrics.')


args = parser.parse_args()

####### Load yaml #######
yaml_file = args.opt
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = CORUN(**x['network_g'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
model_restoration.load_state_dict(checkpoint['params_ema'])
print("===>Testing using weights: ",args.weights)
model_restoration.to(device)
if device.type == 'cuda':
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir)
inp_files = glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg'))

with torch.no_grad():
    for inp_file_ in tqdm(inp_files):
        if device.type == 'cuda':
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = cv2.imread(inp_file_)
        if args.size:
            img = cv2.resize(img, (args.size, args.size))
        img = np.float32(img)/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).to(device)

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)[0]

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        cv2.imwrite((os.path.join(result_dir, os.path.splitext(os.path.split(inp_file_)[-1])[0]+'.png')), img_as_ubyte(restored))
