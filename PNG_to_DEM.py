import rasterio
from rasterio.transform import from_origin
from PIL import Image
import numpy as np

# PNG画像を読み込み
png_path = "dem_tensor_rgb.png"
img = Image.open(png_path).convert("L")  # グレースケールに変換
dem_array = np.array(img).astype(np.float32)

# 値域を0〜1に戻す（PNGは0〜255に量子化されているため）
dem_array = dem_array / 255.0

# ここで必要なら元のDEMのmin/maxを使ってスケーリングを復元
# 例: dem_array = dem_array * (dem_max - dem_min) + dem_min

# GeoTIFFとして保存
tif_path = "restored_dem.tif"
height, width = dem_array.shape

# 仮の座標系・解像度（元DEMの情報がある場合はそれを使う）
transform = from_origin(0, 0, 1, 1)  # 左上座標(0,0)、ピクセルサイズ(1,1)

with rasterio.open(
    tif_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=dem_array.dtype,
    crs="EPSG:4326",  # 仮の座標系（元DEMのCRSを使うべき）
    transform=transform,
) as dst:
    dst.write(dem_array, 1)

print("GeoTIFF DEM saved:", tif_path)
