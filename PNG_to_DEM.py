import rasterio
from rasterio.transform import from_origin
from PIL import Image
import numpy as np

# ---- パス設定 ----
png_path = "dem_tensor_rgb.png"          # 復元元PNG
original_dem_path = "center_tile.tif"   # 元DEM
output_tif_path = "restored_dem.tif"     # 出力ファイル

# ---- 元DEMを読み込む ----
with rasterio.open(original_dem_path) as src:
    original_dem = src.read(1).astype(np.float32)
    dem_min = np.nanmin(original_dem)
    dem_max = np.nanmax(original_dem)
    original_transform = src.transform
    original_crs = src.crs
    height = src.height
    width = src.width

print("Original DEM min/max:", dem_min, dem_max)

# ---- PNG を読み込む（グレースケール前提）----
img = Image.open(png_path).convert("L")
png_array = np.array(img).astype(np.float32)

# サイズチェック（違っていたらエラー）
if png_array.shape != (height, width):
    raise ValueError("PNG と元DEMのサイズが一致していません！")

# ---- 0〜1 に戻す（255量子化されていた前提）----
norm = png_array / 255.0

# ---- 元DEMのスケールに復元 ----
restored_dem = norm * (dem_max - dem_min) + dem_min

# ---- GeoTIFF として保存（元DEMのCRS/transformを引き継ぐ）----
with rasterio.open(
    output_tif_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=restored_dem.dtype,
    crs=original_crs,
    transform=original_transform,
) as dst:
    dst.write(restored_dem, 1)

print("Restored DEM saved:", output_tif_path)

