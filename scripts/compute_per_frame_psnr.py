"""
Compute per-frame RGB PSNR between an original and a reconstructed PNG frame
sequence, and save both the average and the full per-frame list as JSON.

Usage:
    python compute_per_frame_psnr.py \
        --orig-dir path/to/original/frames --orig-pattern "im{i:05d}.png" --orig-start 1 \
        --recon-dir path/to/reconstructed/frames --recon-pattern "recon_frame_{i}.png" --recon-start 0 \
        --num-frames 16 \
        --out per_frame_results.json

The two directories are allowed to use different filename conventions (e.g.
DCVC's test script saves "recon_frame_0.png" with no zero-padding starting at
0, while ffmpeg's PNG extraction saves "im00001.png" zero-padded starting at
1) as long as frame N in one directory corresponds to frame N in the other.
"""
import argparse
import json
import numpy as np
from PIL import Image


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir", required=True)
    p.add_argument("--orig-pattern", default="im{i:05d}.png")
    p.add_argument("--orig-start", type=int, default=1)
    p.add_argument("--recon-dir", required=True)
    p.add_argument("--recon-pattern", default="im{i:05d}.png")
    p.add_argument("--recon-start", type=int, default=1)
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    per_frame = []
    for k in range(args.num_frames):
        orig_path = f"{args.orig_dir}/{args.orig_pattern.format(i=args.orig_start + k)}"
        recon_path = f"{args.recon_dir}/{args.recon_pattern.format(i=args.recon_start + k)}"
        value = psnr(load_rgb(orig_path), load_rgb(recon_path))
        per_frame.append({"frame": k, "psnr": value})
        print(f"frame {k}: PSNR = {value:.3f} dB")

    avg = float(np.mean([f["psnr"] for f in per_frame]))
    print(f"\naverage PSNR = {avg:.3f} dB")

    with open(args.out, "w") as f:
        json.dump({"avg_rgb_psnr": avg, "per_frame": per_frame}, f, indent=2)


if __name__ == "__main__":
    main()
