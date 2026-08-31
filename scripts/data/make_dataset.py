import os
import numpy as np
import pandas as pd
from PIL import Image
import io

def main():
    os.makedirs('RIDCP/rgb_500', exist_ok=True)
    os.makedirs('RIDCP/depth_500', exist_ok=True)
    
    print("Downloading Parquet...")
    url = "https://huggingface.co/datasets/vikhyatk/nyu_depth_v2/resolve/main/data/train-00000-of-00002-99dafa0640c5f4b5.parquet"
    df = pd.read_parquet(url)
    
    print("Extracting 500 images...")
    for i in range(500):
        # image and depth_map are stored as bytes (PNG) in the parquet
        img_bytes = df.iloc[i]['image']['bytes']
        depth_bytes = df.iloc[i]['depth_map']['bytes']
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(f'RIDCP/rgb_500/{i:04d}.png')
        
        depth_img = Image.open(io.BytesIO(depth_bytes))
        depth_array = np.array(depth_img, dtype=np.float32)
        np.save(f'RIDCP/depth_500/{i:04d}.npy', depth_array)

if __name__ == '__main__':
    main()
