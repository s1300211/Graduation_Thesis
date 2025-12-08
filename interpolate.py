import rasterio
import numpy as np
import os
from scipy.interpolate import griddata
# inpaint_biharmonicがscikit-imageのどのバージョンでも動作するように
# inpaintモジュールから関数をインポート
from skimage.restoration import inpaint 

def interpolate_dem_nodata(file_path, output_path, method='inpaint'):
    """
    GeoTIFF DEMのNoData値を補完し、新しいGeoTIFFとして保存する関数。

    Args:
        file_path (str): 入力DEMファイルパス。
        output_path (str): 補完後のDEMを保存するファイルパス。
        method (str): 補完方法 ('nearest' または 'inpaint')。
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません -> {file_path}")
        return

    try:
        with rasterio.open(file_path) as src:
            # メタデータを取得
            profile = src.profile
            nodata = src.nodata
            
            # DEMデータをNumPy配列として読み込み
            dem_data = src.read(1)
            
            # 欠損値マスクの作成
            mask = (dem_data == nodata)
            
            if not np.any(mask):
                print("欠損値 (NoData) は見つかりませんでした。補完はスキップします。")
                return
            
            print(f"欠損値ピクセル数: {np.sum(mask)} ({np.sum(mask) / dem_data.size * 100:.2f}%)")
            print(f"補完方法: {method} を使用して補完を開始します...")

            if method == 'nearest':
                # --- 1. SciPyのgriddataによる近隣補間 (Nearest Neighbor) ---
                
                # 既知の点 (Known Points) の座標と標高値を取得
                known_y, known_x = np.where(~mask)
                known_points = np.vstack((known_y, known_x)).T
                known_values = dem_data[~mask]
                
                # 補完する点 (Interpolation Points) の座標を取得
                interp_y, interp_x = np.where(mask)
                interp_points = np.vstack((interp_y, interp_x)).T

                # griddataで補完
                interpolated_values = griddata(
                    known_points, 
                    known_values, 
                    interp_points, 
                    method='nearest'
                )
                
                filled_data = dem_data.copy()
                filled_data[mask] = interpolated_values
            
            elif method == 'inpaint':
                # --- 2. scikit-imageのinpaintによる画像修復アルゴリズム ---
                
                # V0.16より古いバージョンでは'multichannel=False'が必要でしたが、
                # 新しいバージョンではこの引数は削除されました。
                # 環境に合わせて引数を削除し、修正しました。
                filled_data = inpaint.inpaint_biharmonic(dem_data, mask)
            
            else:
                print("エラー: サポートされていない補完方法です。'nearest' または 'inpaint' を指定してください。")
                return

            # 補完後のDEMを新しいファイルとして保存
            
            # 欠損値はなくなったため、保存時にNoData値を None に設定
            profile.update(nodata=None) 
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(filled_data, 1)

            print(f"補完が完了しました。ファイルが保存されました -> {output_path}")

    except rasterio.RasterioIOError as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました。詳細: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

# --- 実行例 ---
input_file = '/home/s1300211/Documents/datasets/training/NAC_DTM_A17_1500.tif' 
output_file_nearest = 'NAC_DTM_S_Filled_Nearest.tif'
output_file_inpaint = 'NAC_DTM_2000_Filled_Inpaint.tif'

# 補完方法：画像修復 (Inpaint - Bi-harmonic)
interpolate_dem_nodata(input_file, output_file_inpaint, method='inpaint')

# --- 注意 ---
# 実行する際は、上記 'input_file' を正しいファイルパスに書き換え、
# 実行したい行頭の '#' を削除して実行してください。