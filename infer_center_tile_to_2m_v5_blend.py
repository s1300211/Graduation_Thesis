#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import utils


def resample_to_res(src_arr, src_profile, target_res):
    src_transform = src_profile["transform"]
    src_crs = src_profile["crs"]
    src_px = abs(src_transform.a)

    scale = src_px / target_res
    out_h = int(round(src_profile["height"] * scale))
    out_w = int(round(src_profile["width"] * scale))

    out_transform = Affine(
        target_res, src_transform.b, src_transform.c,
        src_transform.d, -target_res, src_transform.f
    )

    dst = np.empty((out_h, out_w), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=out_transform,
        dst_crs=src_crs,
        resampling=Resampling.bilinear,
    )

    out_profile = src_profile.copy()
    out_profile.update(height=out_h, width=out_w, transform=out_transform)
    return dst, out_profile


def make_full_grid_coord(h, w, device):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coord = torch.stack([yy, xx], dim=-1).view(1, h * w, 2)  # (1,N,2)
    global_coord = coord.clone()

    cell = torch.empty((1, h * w, 2), device=device, dtype=torch.float32)
    cell[:, :, 0] = 2.0 / h
    cell[:, :, 1] = 2.0 / w
    return coord, global_coord, cell


def hann2d(h, w):
    # 端が0になる窓。オーバーラップ合成で継ぎ目が消える
    wy = np.hanning(h).astype(np.float32) if h > 2 else np.ones(h, np.float32)
    wx = np.hanning(w).astype(np.float32) if w > 2 else np.ones(w, np.float32)
    w2 = wy[:, None] * wx[None, :]
    # 端が完全0だと重み合計が0になる箇所が出るので少し底上げ
    return np.clip(w2, 1e-3, 1.0)


def down2x_mean(sr2m):
    h, w = sr2m.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    x = sr2m[:h2, :w2]
    return x.reshape(h2//2, 2, w2//2, 2).mean(axis=(1,3))


def affine_calibrate(sr2m, lr4m):
    sr4 = down2x_mean(sr2m)
    h = min(sr4.shape[0], lr4m.shape[0])
    w = min(sr4.shape[1], lr4m.shape[1])
    x = sr4[:h, :w].reshape(-1).astype(np.float64)
    y = lr4m[:h, :w].reshape(-1).astype(np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


@torch.no_grad()
def infer_full_blend(model, lr4_m, device, save_dir="/tmp",
                     inp_size=16, scale=2, stride=8,
                     train_inp_mean=0.15605156123638153,
                     train_inp_std=0.7343724966049194,
                     eps=1e-6,
                     do_affine_calib=True):
    H, W = lr4_m.shape
    Hr, Wr = H * scale, W * scale

    acc = np.zeros((Hr, Wr), dtype=np.float32)
    wsum = np.zeros((Hr, Wr), dtype=np.float32)

    hr_patch = inp_size * scale              # 32
    N = hr_patch * hr_patch                  # 1024
    coord, global_coord, cell = make_full_grid_coord(hr_patch, hr_patch, device)

    batch_idx = torch.tensor(0, device=device)
    add_args = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)

    w2 = hann2d(hr_patch, hr_patch)  # (32,32)

    # 端までカバーするため、最後の位置を必ず含める
    ys = list(range(0, max(H - inp_size + 1, 1), stride))
    xs = list(range(0, max(W - inp_size + 1, 1), stride))
    if ys[-1] != H - inp_size:
        ys.append(H - inp_size)
    if xs[-1] != W - inp_size:
        xs.append(W - inp_size)

    for y in ys:
        for x in xs:
            patch_m = lr4_m[y:y + inp_size, x:x + inp_size].astype(np.float32)

            pm = float(patch_m.mean())
            ps = float(patch_m.std())
            if ps < eps:
                ps = eps

            patch_z = (patch_m - pm) / ps
            patch_inp = patch_z * train_inp_std + train_inp_mean

            inp = torch.from_numpy(patch_inp).to(device=device, dtype=torch.float32)[None, None, :, :]
            gt_dummy = torch.zeros((1, N, 1), device=device, dtype=torch.float32)

            batch = {
                "inp": inp,
                "coord": coord,
                "cell": cell,
                "gt": gt_dummy,
                "add_args": add_args,
                "global_coord": global_coord,
            }

            outd = model(batch, batch_idx, flag="test", epoch=0, save_dir=save_dir)
            if not isinstance(outd, dict) or "pred" not in outd:
                raise RuntimeError(f"Model did not return 'pred'. keys={list(outd.keys()) if isinstance(outd, dict) else type(outd)}")

            pred_inp = outd["pred"].view(hr_patch, hr_patch).detach().float().cpu().numpy().astype(np.float32)

            pred_z = (pred_inp - train_inp_mean) / max(train_inp_std, eps)
            pred_m = pred_z * ps + pm

            yy = y * scale
            xx = x * scale

            acc[yy:yy + hr_patch, xx:xx + hr_patch] += pred_m * w2
            wsum[yy:yy + hr_patch, xx:xx + hr_patch] += w2

    out = acc / np.maximum(wsum, 1e-6)

    if do_affine_calib:
        a, b = affine_calibrate(out, lr4_m)
        out = (a * out + b).astype(np.float32)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr5_tif", required=True)
    ap.add_argument("--out_tif", required=True)
    ap.add_argument("--model_pth", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_dir", default="/tmp")
    ap.add_argument("--stride", type=int, default=8)  # ★格子消しの要
    ap.add_argument("--train_inp_mean", type=float, default=0.15605156123638153)
    ap.add_argument("--train_inp_std", type=float, default=0.7343724966049194)
    ap.add_argument("--no_affine_calib", action="store_true")
    args = ap.parse_args()

    device = args.device

    ck = torch.load(args.model_pth, map_location="cpu")
    mspec = ck["model"].copy()
    sd = mspec.pop("sd")
    model = utils.object_from_dict(mspec).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    with rasterio.open(args.lr5_tif) as src:
        lr5 = src.read(1).astype(np.float32)
        profile5 = src.profile

    lr4, profile4 = resample_to_res(lr5, profile5, target_res=4.0)

    out_dem = infer_full_blend(
        model, lr4, device=device, save_dir=args.save_dir,
        stride=args.stride,
        train_inp_mean=args.train_inp_mean,
        train_inp_std=args.train_inp_std,
        do_affine_calib=(not args.no_affine_calib)
    )

    profile2 = profile4.copy()
    t4 = profile4["transform"]
    t2 = Affine(2.0, t4.b, t4.c, t4.d, -2.0, t4.f)
    profile2.update(
        height=out_dem.shape[0],
        width=out_dem.shape[1],
        transform=t2,
        dtype="float32",
        count=1,
        compress="LZW",
    )

    with rasterio.open(args.out_tif, "w", **profile2) as dst:
        dst.write(out_dem.astype(np.float32), 1)

    print("[DONE] wrote:", args.out_tif)


if __name__ == "__main__":
    main()
