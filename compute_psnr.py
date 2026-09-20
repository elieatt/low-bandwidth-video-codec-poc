import sys
import numpy as np
from PIL import Image

orig_dir = sys.argv[1]
recon_dir = sys.argv[2]
n = int(sys.argv[3])

psnrs = []
for i in range(1, n + 1):
    o = np.asarray(Image.open(f"{orig_dir}/im{i:05d}.png").convert("RGB")).astype(np.float64) / 255
    r = np.asarray(Image.open(f"{recon_dir}/im{i:05d}.png").convert("RGB")).astype(np.float64) / 255
    mse = np.mean((o - r) ** 2)
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    psnrs.append(psnr)
    print(f"frame {i}: PSNR = {psnr:.3f} dB")

print(f"\naverage PSNR = {np.mean(psnrs):.3f} dB")
