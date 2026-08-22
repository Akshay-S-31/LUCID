import os
import numpy as np
from datasets import load_dataset
from PIL import Image

def main():
    print("Creating directories...")
    rgb_dir = 'Datasets/RIDCP/rgb_500'
    depth_dir = 'Datasets/RIDCP/depth_500'
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print("Downloading NYU Depth V2 from HuggingFace (this might take a minute)...")
    # Using the sayakpaul version which is properly formatted for PyTorch/PIL
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train", trust_remote_code=True)

    print("Saving 500 RGB images and .npy Depth maps...")
    # The dataset has 1449 pairs. We just need 500.
    for i in range(500):
        example = dataset[i]
        
        # Save RGB Image as PNG
        rgb = example['image']
        rgb_path = os.path.join(rgb_dir, f"{i:04d}.png")
        rgb.save(rgb_path)
        
        # Save Depth Map as .npy array 
        # (I checked the CORUN dataloader code, and it specifically looks for .npy files!)
        depth_img = example['depth_map']
        depth_array = np.array(depth_img, dtype=np.float32)
        depth_path = os.path.join(depth_dir, f"{i:04d}.npy")
        np.save(depth_path, depth_array)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/500 images...")
            
    print("\n✅ Substitute Dataset downloaded successfully!")
    print(f"RGB images saved to: {rgb_dir}")
    print(f"Depth maps saved to: {depth_dir}")
    print("You can now safely skip the Baidu Pan download!")

if __name__ == '__main__':
    main()
