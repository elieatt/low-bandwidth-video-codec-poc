"""
Run MLVC (https://github.com/microsoft/mlvc) on a raw YUV420 video and save
every reconstructed frame plus per-frame PSNR/bitrate to a JSON file.

This script must be placed inside the mlvc repo's `video/` directory (or run
with that directory on PYTHONPATH), since it imports MLVC's own `src` package.
It is not a standalone package, it's a thin driver around MLVC's own model and
entropy coder, adapted from MLVC's own `video/notebooks/demo.ipynb`.

Setup (once, from the mlvc repo root):
    uv sync --extra onnxruntime
    uv pip install packages/msrtc_rans   # needs a C++ compiler, see repo README

Usage (from mlvc/video/, or with mlvc/video on PYTHONPATH):
    uv run python run_mlvc.py \
        --checkpoint /path/to/mlvc-psnr-v1.ckpt \
        --video /path/to/clip_320x256.yuv \
        --width 320 --height 256 \
        --q-index 21 \
        --out-dir ./out

Checkpoints and their SHA-256 hashes are listed at
https://github.com/microsoft/mlvc#models
"""
import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.utils.video_reader import YUVReader
from src.utils.model_factory import create_video_model
from src.utils.stream_helper import get_state_dict, prepare_frame
from src.transforms.functional import yuv_444_to_420, yuv_420_to_444, ycbcr2rgb

MODEL_PARAMS = {
    "type": "DMC-6.1sb",
    "activation": "LeakyReLU",
    "input_offset": -0.5,
    "feature_channels": 128,
    "spatial_prior_channels": 256,
    "memory_activation": "identity",
    "zero_init_residual": True,
    "chunk_mode": "gated",
    "ffn_gate_activation": "ReLU1",
    "chain_feature_adaptors": True,
}


def read_yuv420_frames(video_path, width, height):
    video_reader = YUVReader(str(video_path), width, height)
    res = []
    while True:
        y, uv = video_reader.read_one_frame()
        if y is None:
            break
        res.append((y, uv))
    return res


def load_model(checkpoint_path):
    state_dict = get_state_dict(checkpoint_path)
    model = create_video_model(config=MODEL_PARAMS)
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model


def yuv420_to_rgb(yuv420):
    y = torch.from_numpy(yuv420[0]).float().unsqueeze(0)
    u, v = torch.from_numpy(yuv420[1]).float().unsqueeze(0).chunk(2, dim=1)
    yuv444 = yuv_420_to_444((y, u, v), mode="nearest")
    rgb = ycbcr2rgb(yuv444[0]).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((np.round(rgb * 255.0)).astype(np.uint8))


def prepare_output(x_hat, padding):
    x_hat_cropped = torch.nn.functional.pad(x_hat, tuple(-x for x in padding))
    y_rec, u_rec, v_rec = yuv_444_to_420(x_hat_cropped.to(torch.float32))
    uv_rec = torch.cat((u_rec, v_rec), dim=1)
    return (y_rec.cpu().numpy()[0], uv_rec.cpu().numpy()[0])


def calc_rgb_psnr(img_a, img_b):
    a = np.asarray(img_a).astype(np.float64) / 255
    b = np.asarray(img_b).astype(np.float64) / 255
    mse = np.mean((a - b) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse))


class MlVideoCodec:
    """Minimal encode loop, adapted from mlvc/video/notebooks/demo.ipynb."""

    def __init__(self, model, reset_period=64):
        self._model = model
        self._dpb = None
        self._frame_num = 0
        self._reset_period = reset_period

    @property
    def fa_idx(self):
        frame_index_map = self._model.frame_index_map
        return frame_index_map[self._frame_num % len(frame_index_map)]

    @torch.inference_mode()
    def encode(self, yuv420, q_index):
        device = next(self._model.parameters()).device
        x, _, padding = prepare_frame(yuv420, is_yuv420=True, precision="fp32", device=device)

        if self._dpb is None:
            self._dpb = dict(ref_frame=torch.ones_like(x) * 0.5, ref_feature=None)

        if self._reset_period is not None and self._frame_num % self._reset_period == 1:
            self._dpb["ref_feature"] = None

        result = self._model.compress(x, dpb=self._dpb, q_index=q_index, fa_idx=self.fa_idx, calc_bits_estimates=False)
        self._dpb = result["dpb"]
        self._frame_num += 1

        return result["bit_stream"], prepare_output(result["x_hat"], padding)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True, help="raw YUV420 file")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--q-index", type=int, default=42, help="0-63, higher = more bitrate/quality")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-frames", action="store_true", help="save every reconstructed PNG frame")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model = load_model(args.checkpoint)

    print("reading frames...")
    frames = read_yuv420_frames(args.video, args.width, args.height)
    print(f"{len(frames)} frames loaded")

    codec = MlVideoCodec(model)
    total_bits = 0
    per_frame = []
    for i, yuv420_inp in enumerate(frames):
        bit_stream, yuv420_rec = codec.encode(yuv420_inp, q_index=args.q_index)
        total_bits += 8 * len(bit_stream)

        img_inp = yuv420_to_rgb(yuv420_inp)
        img_rec = yuv420_to_rgb(yuv420_rec)
        psnr = calc_rgb_psnr(img_inp, img_rec)
        per_frame.append({"frame": i, "psnr": psnr, "bits": 8 * len(bit_stream)})
        print(f"frame {i}: psnr={psnr:.2f} dB, bits={8 * len(bit_stream)}")

        if args.save_frames:
            img_rec.save(out_dir / f"recon_frame_{i:05d}.png")
            if i == 0:
                img_inp.save(out_dir / "orig_frame_00000.png")

    n_pixels = args.width * args.height * len(frames)
    bpp = total_bits / n_pixels
    kbps = bpp * args.width * args.height * args.fps / 1000

    summary = {
        "q_index": args.q_index,
        "avg_rgb_psnr": float(np.mean([f["psnr"] for f in per_frame])),
        "bpp": bpp,
        "kbps": kbps,
        "per_frame": per_frame,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\naverage PSNR = {summary['avg_rgb_psnr']:.3f} dB, ~{kbps:.1f} kbps")


if __name__ == "__main__":
    main()
