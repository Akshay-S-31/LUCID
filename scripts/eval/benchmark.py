import sys
import os

# Crucial: This explicitly tells Python to look inside the corun_colabator folder for internal imports like 'utils'
sys.path.append(os.path.abspath('corun_colabator'))

import yaml
import time
import torch
from corun_colabator.archs.corun_arch import CORUN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_model(name, weights_path):
    opt = yaml.safe_load(open('dehazing_options/valid_corun.yml', 'r'))
    opt['network_g'].pop('type')
    model = CORUN(**opt['network_g']).to(device)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['params_ema'], strict=False)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy_input = torch.randn(1, 3, 1080, 1920).to(device)
    
    with torch.no_grad():
        for _ in range(5): model(dummy_input)
        
    start = time.time()
    with torch.no_grad():
        for _ in range(20): model(dummy_input)
    fps = 20 / (time.time() - start)
    
    print(f'\n{name}:')
    print(f'  Parameters: {params:.2f} M')
    print(f'  Speed:      {fps:.2f} FPS (on 1080p)')

print('Running highly-precise FPS and Parameter benchmark...')
benchmark_model('Baseline CORUN+', './CORUN+.pth')
benchmark_model('Ours (5k iter)', './net_g_5000.pth')
