# DEMの中心の500x500タイルを切り取って保存するスクリプト
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# 入力ファイルパス
input_path = "/home/s1300211/Documents/m5261147/Data/NAC_DTM_equatorial/NAC_DTM_A17SIVB.tiff"
# 出力ファイルパス
output_path = "NAC_DTM_1500.tif"
# タイルサイズ（ピクセル）
tile_size = 1500

with rasterio.open(input_path) as src:
    # 全体のサイズ
    width = src.width
    height = src.height

    # 中心座標（ピクセル単位）
    center_x = width // 2
    center_y = height // 2

    # 切り出し範囲（左上座標）
    left = center_x - tile_size // 2
    top = center_y - tile_size // 2

    # ウィンドウ定義
    window = Window(left, top, tile_size, tile_size)

    # データ読み込み（全バンド）
    data = src.read(window=window)

    # 地理変換情報の更新
    transform = src.window_transform(window)

    # 出力ファイル作成
    profile = src.profile
    profile.update({
        "height": tile_size,
        "width": tile_size,
        "transform": transform
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

print("中心タイルを保存しました:", output_path)
