"""
Run Microsoft's DCVC (https://github.com/microsoft/DCVC) on a folder of PNG
frames at a given quality preset. This is a thin wrapper around DCVC's own
`test_video.py`: it writes the dataset config JSON that script expects and
invokes it with the right checkpoint paths, so you don't have to hand-write
JSON or remember which I-frame checkpoint pairs with which quality preset.

Setup (once): follow steps 1-3 in the main README to clone DCVC and download
its checkpoints into DCVC/DCVC-family/DCVC/checkpoints/.

Usage:
    python run_dcvc.py \
        --dcvc-dir /path/to/DCVC/DCVC-family/DCVC \
        --frames-dir /path/to/testdata/akiyo_320x256 \
        --sequence-name akiyo_320x256 \
        --num-frames 16 --gop 16 \
        --quality 0 \
        --out-prefix akiyo_q0

--frames-dir should be the parent of a folder named --sequence-name containing
im00001.png, im00002.png, ... (DCVC's own expected layout).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

QUALITY_TO_IFRAME_CKPT = {
    0: "cheng2020-anchor-3-e49be189.pth.tar",
    1: "cheng2020-anchor-4-98b0b468.pth.tar",
    2: "cheng2020-anchor-5-23852949.pth.tar",
    3: "cheng2020-anchor-6-4c052b1a.pth.tar",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dcvc-dir", required=True, help="path to DCVC/DCVC-family/DCVC")
    p.add_argument("--frames-dir", required=True, help="parent dir containing the sequence folder")
    p.add_argument("--sequence-name", required=True, help="subfolder of --frames-dir with the PNG frames")
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--gop", type=int, default=16)
    p.add_argument("--quality", type=int, choices=[0, 1, 2, 3], required=True)
    p.add_argument("--out-prefix", required=True, help="prefix for result/recon output files")
    args = p.parse_args()

    dcvc_dir = Path(args.dcvc_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve()

    config = {
        args.out_prefix: {
            "base_path": os.path.relpath(frames_dir, dcvc_dir).replace("\\", "/"),
            "sequences": {
                args.sequence_name: {"frames": args.num_frames, "gop": args.gop}
            },
        }
    }
    config_path = dcvc_dir / f"{args.out_prefix}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"wrote {config_path}")

    i_frame_ckpt = f"checkpoints/{QUALITY_TO_IFRAME_CKPT[args.quality]}"
    model_ckpt = f"checkpoints/model_dcvc_quality_{args.quality}_psnr.pth"
    result_json = f"{args.out_prefix}_result.json"
    recon_dir = f"{args.out_prefix}_recon"

    cmd = [
        sys.executable, "test_video.py",
        "--i_frame_model_name", "cheng2020-anchor",
        "--i_frame_model_path", i_frame_ckpt,
        "--test_config", config_path.name,
        "--cuda", "false",
        "--worker", "1",
        "--output_json_result_path", result_json,
        "--model_type", "psnr",
        "--recon_bin_path", recon_dir,
        "--write_recon_frame", "true",
        "--write_stream", "false",
        "--model_path", model_ckpt,
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=dcvc_dir, check=True)

    with open(dcvc_dir / result_json) as f:
        result = json.load(f)
    stats = result[args.out_prefix][args.sequence_name][f"model_dcvc_quality_{args.quality}_psnr.pth"]
    bpp = stats["ave_all_frame_bpp"]
    psnr = stats["ave_all_frame_quality"]
    print(f"\nresult: {psnr:.3f} dB at {bpp:.6f} bpp")
    print(f"reconstructed frames: {dcvc_dir / recon_dir}")
    print(f"full result JSON: {dcvc_dir / result_json}")


if __name__ == "__main__":
    main()
