"""
Compose N images side by side with a label (and optional stats line) above
each one. Used to build the "original vs codec A vs codec B..." comparison
images in this repo.

Usage:
    python build_comparison_image.py \
        --out comparison.png \
        --panel "Original" "" path/to/original.png \
        --panel "MLVC" "32.4 dB, 38 kbps" path/to/mlvc_recon.png \
        --panel "DCVC" "31.7 dB, 42 kbps" path/to/dcvc_recon.png \
        --panel "H.264" "23.6 dB, 41 kbps" path/to/h264_recon.png

Each --panel takes exactly three values: label, stats (pass "" for none), and
image path. Repeat --panel for as many images as you want side by side.
"""
import argparse
from PIL import Image, ImageDraw, ImageFont


def find_font(size):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", nargs=3, action="append", required=True, metavar=("LABEL", "STATS", "IMAGE_PATH"))
    p.add_argument("--out", required=True)
    p.add_argument("--gap", type=int, default=10)
    p.add_argument("--header-height", type=int, default=40)
    args = p.parse_args()

    imgs = [Image.open(path).convert("RGB") for _, _, path in args.panel]
    w, h = imgs[0].size
    n = len(imgs)

    combo = Image.new("RGB", (w * n + args.gap * (n - 1), h + args.header_height), "white")
    draw = ImageDraw.Draw(combo)
    font = find_font(15)
    font_small = find_font(12)

    x = 0
    for (label, stats, _), img in zip(args.panel, imgs):
        combo.paste(img, (x, args.header_height))
        draw.text((x + 6, 8), label, fill="black", font=font)
        if stats:
            draw.text((x + 6, 26), stats, fill="#555555", font=font_small)
        x += w + args.gap

    combo.save(args.out)
    print(f"saved {args.out} ({combo.size[0]}x{combo.size[1]})")


if __name__ == "__main__":
    main()
