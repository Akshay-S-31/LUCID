import os
import argparse
import pyiqa
from tqdm import tqdm
import torch

os.environ['HF_HUB_OFFLINE'] = 'True'

parser = argparse.ArgumentParser(description='Evaluate Image Quality (Full Metrics)')
parser.add_argument('--input_dir', required=True, type=str, help='Directory of dehazed images')
parser.add_argument('--gt_dir', required=True, type=str, help='Directory of Ground Truth images')
args = parser.parse_args()

print("Loading PyIQA Metrics...")
brisque = pyiqa.create_metric('brisque')
nima = pyiqa.create_metric('nima')
psnr = pyiqa.create_metric('psnr')
ssim = pyiqa.create_metric('ssim')
lpips = pyiqa.create_metric('lpips')

files = os.listdir(args.input_dir)

sum_brisque = 0
sum_nima = 0
sum_psnr = 0
sum_ssim = 0
sum_lpips = 0
count = 0

for file in tqdm(files):
    if file.endswith('Store') or file.endswith('.txt'):
        continue
    
    input_img = os.path.join(args.input_dir, file)
    
    # Map input file name to GT file name if needed (e.g. ih31_hazy.png -> ih31.png)
    gt_file = file.replace('_hazy', '')
    gt_img = os.path.join(args.gt_dir, gt_file)
    
    # Try .jpg and .JPG for O-Haze
    if not os.path.exists(gt_img):
        gt_img = os.path.join(args.gt_dir, gt_file.replace('.png', '.jpg'))
    if not os.path.exists(gt_img):
        gt_img = os.path.join(args.gt_dir, gt_file.replace('.png', '.JPG'))
    
    if os.path.exists(gt_img):
        # Create a temporary resized GT to match the 256x256 fast inference output
        import cv2
        gt_cv = cv2.imread(gt_img)
        gt_cv_resized = cv2.resize(gt_cv, (256, 256))
        temp_gt = "temp_gt.png"
        cv2.imwrite(temp_gt, gt_cv_resized)

        # Calculate blind metrics
        dist_brisque = brisque(input_img)
        dist_nima = nima(input_img)
        
        # Calculate reference metrics using the resized GT
        dist_psnr = psnr(input_img, temp_gt)
        dist_ssim = ssim(input_img, temp_gt)
        dist_lpips = lpips(input_img, temp_gt)
        
        sum_brisque += dist_brisque.item() if torch.is_tensor(dist_brisque) else dist_brisque
        sum_nima += dist_nima.item() if torch.is_tensor(dist_nima) else dist_nima
        sum_psnr += dist_psnr.item() if torch.is_tensor(dist_psnr) else dist_psnr
        sum_ssim += dist_ssim.item() if torch.is_tensor(dist_ssim) else dist_ssim
        sum_lpips += dist_lpips.item() if torch.is_tensor(dist_lpips) else dist_lpips
        
        count += 1
    else:
        print(f"Warning: GT not found for {file} (expected {gt_img})")

print(f'\n--- Evaluation Results for {args.input_dir} ---')
if count > 0:
    print(f'Evaluated {count} paired images.')
    print(f'Average PSNR:    {sum_psnr/count:.4f}')
    print(f'Average SSIM:    {sum_ssim/count:.4f}')
    print(f'Average LPIPS:   {sum_lpips/count:.4f}')
    print(f'Average BRISQUE: {sum_brisque/count:.4f}')
    print(f'Average NIMA:    {sum_nima/count:.4f}')
else:
    print('No paired images were found.')
