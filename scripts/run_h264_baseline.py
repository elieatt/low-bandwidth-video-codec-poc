"""
Encode a PNG frame sequence with H.264 at a target constant bitrate, decode it
back to PNGs, and report the actual achieved bitrate (which will differ
slightly from your target, use the reported one for a fair comparison against
a neural codec's own bpp/kbps figure).

Requires ffmpeg with libx264 support on PATH.

Usage:
    python run_h264_baseline.py \
        --frames-glob "testdata/akiyo_320x256/akiyo_320x256/im%05d.png" \
        --fps 29.97 --gop 16 \
        --target-kbps 42 \
        --out-video akiyo_h264_42k.mp4 \
        --recon-dir h264_recon_42k
"""
import argparse
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-glob", required=True, help="ffmpeg input pattern, e.g. dir/im%%05d.png")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--gop", type=int, default=16)
    p.add_argument("--target-kbps", type=float, required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--recon-dir", required=True)
    p.add_argument("--bufsize-kbps", type=float, default=10,
                    help="VBV buffer size in kbps. This repo's own runs used 10 (or "
                         "occasionally 5-15) throughout, not a value relative to the "
                         "target, since on a clip this short the buffer size measurably "
                         "shifts the actual achieved bitrate, use the same value to "
                         "reproduce a specific published number exactly")
    args = p.parse_args()

    br = f"{args.target_kbps:.0f}k"
    bufsize = f"{args.bufsize_kbps:.0f}k"

    encode_cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps), "-i", args.frames_glob,
        "-c:v", "libx264", "-x264-params", "nal-hrd=cbr",
        "-b:v", br, "-minrate", br, "-maxrate", br, "-bufsize", bufsize,
        "-g", str(args.gop), "-pix_fmt", "yuv420p",
        args.out_video,
    ]
    print("encoding:", " ".join(encode_cmd))
    result = subprocess.run(encode_cmd, capture_output=True, text=True)
    match = re.search(r"kb/s:\s*([\d.]+)", result.stderr)
    actual_kbps = float(match.group(1)) if match else None
    if actual_kbps is None:
        print(result.stderr[-2000:])
        raise RuntimeError("could not parse achieved bitrate from ffmpeg output")

    subprocess.run(["mkdir" if False else "python", "-c", f"import os; os.makedirs(r'{args.recon_dir}', exist_ok=True)"])
    decode_cmd = ["ffmpeg", "-y", "-i", args.out_video, "-pix_fmt", "rgb24", f"{args.recon_dir}/im%05d.png"]
    print("decoding:", " ".join(decode_cmd))
    subprocess.run(decode_cmd, capture_output=True, text=True)

    print(f"\ntarget: {args.target_kbps:.1f} kbps, actual encoded: {actual_kbps:.2f} kbps")
    print(f"reconstructed frames: {args.recon_dir}")
    print("use the ACTUAL kbps above, not your target, when comparing PSNR at 'matched bitrate'.")


if __name__ == "__main__":
    main()
