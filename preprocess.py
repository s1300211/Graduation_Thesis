import rasterio
import numpy as np
import torch
from torchvision.utils import save_image

# DEMファイルを読み込み（例: GeoTIFF）
dem_path = "center_tile.tif"
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)  # 1チャンネルDEMをfloat32で取得

# 値域を正規化（例: min-max正規化）
# DEMの範囲に応じて調整してください
dem_min, dem_max = np.min(dem), np.max(dem)
dem_norm = (dem - dem_min) / (dem_max - dem_min + 1e-8)

# PyTorch tensorに変換
tensor = torch.from_numpy(dem_norm).unsqueeze(0).unsqueeze(0)  
# shape: (batch=1, channel=1, height, width)

# SROOEはRGB(3ch)を想定 → チャンネルを複製
tensor_rgb = tensor.repeat(1, 3, 1, 1)  
# shape: (batch=1, channel=3, height, width)

print("Input tensor shape:", tensor_rgb.shape)
print("Value range:", tensor_rgb.min().item(), tensor_rgb.max().item())

save_image(tensor_rgb, "/home/s1300211/Documents/datasets/SR_testing_datasets_crop_x4/Set5_LRx4/dem_tensor_rgb.png")

# これをSROOEモデルに入力可能
# 例: output = srooe_model(tensor_rgb)
