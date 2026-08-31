import pyiqa
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate LPIPS')
parser.add_argument('--input_dir', required=True, type=str)
parser.add_argument('--gt_dir', required=True, type=str)
args = parser.parse_args()

lpips = pyiqa.create_metric('lpips')

files = os.listdir(args.input_dir)
sum_lpips = 0
count = 0

for file in tqdm(files):
    if file.endswith('Store') or file.endswith('.txt'):
        continue
    input_img = os.path.join(args.input_dir, file)
    gt_img = os.path.join(args.gt_dir, file)
    if os.path.exists(gt_img):
        sum_lpips += lpips(input_img, gt_img)
        count += 1

print(f'\n{args.input_dir}')
print(f'Average LPIPS: {(sum_lpips/count).item():.4f} ({count} images)')
