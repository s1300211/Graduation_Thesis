import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def display_dem(file_path):
    """
    GeoTIFF形式のDEMファイル（標高データ）を読み込み、カラーマップで表示する関数。
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません -> {file_path}")
        return

    try:
        with rasterio.open(file_path) as src:
            # 1. データの読み込み
            # DEMデータは通常、最初のバンド（インデックス1）に格納されています。
            # float32データ（標高値）をNumPy配列として読み込みます。
            dem_data = src.read(1)
            
            # 2. NoData値の処理
            # NoData値（-3.4e+38）は表示上不要なため、NaN（非数）に置き換えます。
            nodata_value = src.nodata
            if nodata_value is not None:
                dem_data[dem_data == nodata_value] = np.nan
            
            # 3. データの可視化
            
            # 論文のデータは月の南極周辺（極座標）であり、標高値の範囲が広いと予想されます。
            # 適切な標高範囲（vmin, vmax）を設定することで、地形の起伏を見やすくします。
            # ここでは、データの平均値 ± 2標準偏差を範囲の目安とします。（外れ値の影響を減らすため）
            valid_data = dem_data[~np.isnan(dem_data)]
            if valid_data.size > 0:
                mean_elev = valid_data.mean()
                std_elev = valid_data.std()
                vmin = mean_elev - 2 * std_elev
                vmax = mean_elev + 2 * std_elev
            else:
                vmin, vmax = np.nanmin(dem_data), np.nanmax(dem_data)


            # プロットの設定
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # imshow関数でデータを表示します。
            # cmap='terrain' は地形データを表示するのによく使われるカラーマップです。
            image = ax.imshow(dem_data, cmap='terrain', vmin=vmin, vmax=vmax)

            # カラーバーを追加（標高のスケールを示す）
            cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Elevation (m) - 標高 [メートル]')

            # タイトルと軸ラベルの設定
            ax.set_title(f'Visualizing DEM: {os.path.basename(file_path)} (2m/pix)')
            ax.set_xlabel(f"Columns (X-pixel, Resolution: {abs(src.transform.a):.1f}m)")
            ax.set_ylabel(f"Rows (Y-pixel, Resolution: {abs(src.transform.e):.1f}m)")
            
            # 画像の表示
            plt.show()

    except rasterio.RasterioIOError as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました。")
        print(f"詳細: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

# --- 実行例 ---
# 実際に表示したいDEMファイルへのパスを指定してください
dem_file_path = 'gene_test_imge.tif' 

display_dem(dem_file_path)
# 上記のコメントを外して実行してください。