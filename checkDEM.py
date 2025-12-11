import rasterio
import numpy as np
import sys

def calculate_dem_range(file_path):
    """
    指定されたDEMファイルから、非NaN値の最大標高、最小標高、および標高差を計算して表示する。

    Args:
        file_path (str): 処理するDEMファイル（例: GeoTIFF）のパス。
    """
    print(f"--- 📊 ファイル名: {file_path} の統計情報 ---")
    
    try:
        # rasterioを使用してDEMファイルを開く
        with rasterio.open(file_path) as src:
            # データをNumPy配列として読み込む
            data = src.read(1)
            
            # nodata値（NaNとして扱われる可能性のある値）を取得
            nodata_val = src.nodata
            
            # --- NaN値の除外処理 ---
            
            # 1. データの配列を平坦化（1次元配列に変換）する
            flat_data = data.flatten()
            
            # 2. nodata値とNaN値を除外する
            # NumPyのNaN（np.nan）またはデータセットのnodata値を除外します
            
            # まずNaN値を除外（データにnp.nanが含まれる場合）
            valid_data = flat_data[~np.isnan(flat_data)]
            
            # 次に、nodata値を除外（データに明示的なnodata値が設定されている場合）
            if nodata_val is not None:
                valid_data = valid_data[valid_data != nodata_val]
            
            # --- 統計情報の計算 ---
            
            if valid_data.size == 0:
                print("🚫 エラー: 有効な標高データが見つかりませんでした。")
                return

            # 最小標高の計算
            min_elevation = np.min(valid_data)
            
            # 最大標高の計算
            max_elevation = np.max(valid_data)
            
            # 標高差（ダイナミックレンジ）の計算
            elevation_difference = max_elevation - min_elevation
            
            # --- 結果の表示 ---
            
            print(f"✅ 計算に使用した有効なデータ数: {valid_data.size} / {data.size}")
            print(f"➡️ 最小標高 (Min): {min_elevation:,.4f}")
            print(f"⬆️ 最大標高 (Max): {max_elevation:,.4f}")
            print(f"📏 標高差 (Range): {elevation_difference:,.4f}")

    except rasterio.RasterioIOError:
        print(f"❌ エラー: ファイル '{file_path}' が見つからないか、読み込めませんでした。")
    except Exception as e:
        print(f"🛑 予期せぬエラーが発生しました: {e}")

# --- 実行部分 ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # スクリプト実行時に引数としてファイルパスが指定されなかった場合
        print("💡 使用方法: python checkDEM.py <DEMファイルパス>")
        print("例: python calculate_dem.py training_dem_2m.tif")
    else:
        # コマンドライン引数からファイルパスを取得
        dem_file_path = sys.argv[1]
        calculate_dem_range(dem_file_path)