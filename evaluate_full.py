import os
import argparse
import pyiqa
from tqdm import tqdm
import torch

os.environ['HF_HUB_OFFLINE'] = 'True'

parser = argparse.ArgumentParser(description='Evaluate Image Quality (Full Metrics)')
parser.add_argument('--input_dir', required=True, type=str, help='Directory of dehazed images')
parser.add_argument('--gt_dir', required=True, type=str, help='Directory of Ground Truth images')
parser.add_argument('--size', default=0, type=int,
                    help='If non-zero, resize BOTH the result and the ground '
                         'truth to SIZE x SIZE before computing paired metrics. '
                         'Default 0 compares at the ground truth resolution, '
                         'resampling the result to match only if the two differ. '
                         'Any non-zero value changes PSNR/SSIM/LPIPS and makes '
                         'them incomparable with published numbers.')
args = parser.parse_args()

print("Loading PyIQA Metrics...")
brisque = pyiqa.create_metric('brisque')
nima = pyiqa.create_metric('nima')
psnr = pyiqa.create_metric('psnr')
ssim = pyiqa.create_metric('ssim')
lpips = pyiqa.create_metric('lpips')
niqe = pyiqa.create_metric('niqe')
musiq = pyiqa.create_metric('musiq')

files = os.listdir(args.input_dir)

sum_brisque = 0
sum_nima = 0
sum_psnr = 0
sum_ssim = 0
sum_lpips = 0
sum_niqe = 0
sum_musiq = 0
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
        import cv2

        # Paired metrics need both images on the same grid. By default that grid
        # is the ground truth's own resolution: the result is resampled to match
        # the GT only when they differ, and the GT is never degraded. Passing
        # --size forces both onto a SIZE x SIZE grid instead, which is only
        # appropriate when the results were themselves produced at that size.
        res_cv = cv2.imread(input_img)
        gt_cv = cv2.imread(gt_img)

        if args.size:
            res_cv = cv2.resize(res_cv, (args.size, args.size))
            gt_cv = cv2.resize(gt_cv, (args.size, args.size))
        elif res_cv.shape[:2] != gt_cv.shape[:2]:
            res_cv = cv2.resize(res_cv, (gt_cv.shape[1], gt_cv.shape[0]))

        temp_res, temp_gt = "temp_res.png", "temp_gt.png"
        cv2.imwrite(temp_res, res_cv)
        cv2.imwrite(temp_gt, gt_cv)

        # Blind metrics are computed on the result as produced.
        dist_brisque = brisque(input_img)
        dist_nima = nima(input_img)
        dist_niqe = niqe(input_img)
        dist_musiq = musiq(input_img)

        # Reference metrics on the aligned pair.
        dist_psnr = psnr(temp_res, temp_gt)
        dist_ssim = ssim(temp_res, temp_gt)
        dist_lpips = lpips(temp_res, temp_gt)

        val = lambda d: d.item() if torch.is_tensor(d) else d
        sum_brisque += val(dist_brisque)
        sum_nima += val(dist_nima)
        sum_niqe += val(dist_niqe)
        sum_musiq += val(dist_musiq)
        sum_psnr += val(dist_psnr)
        sum_ssim += val(dist_ssim)
        sum_lpips += val(dist_lpips)

        count += 1
    else:
        print(f"Warning: GT not found for {file} (expected {gt_img})")

for _tmp in ('temp_res.png', 'temp_gt.png'):
    if os.path.exists(_tmp):
        os.remove(_tmp)

print(f'\n--- Evaluation Results for {args.input_dir} ---')
if count > 0:
    grid = f'{args.size}x{args.size}' if args.size else 'ground-truth resolution'
    print(f'Evaluated {count} paired images at {grid}.')
    print(f'Average PSNR:    {sum_psnr/count:.4f}')
    print(f'Average SSIM:    {sum_ssim/count:.4f}')
    print(f'Average LPIPS:   {sum_lpips/count:.4f}')
    print(f'Average BRISQUE: {sum_brisque/count:.4f}')
    print(f'Average NIMA:    {sum_nima/count:.4f}')
    print(f'Average NIQE:    {sum_niqe/count:.4f}')
    print(f'Average MUSIQ:   {sum_musiq/count:.4f}')
else:
    print('No paired images were found.')
