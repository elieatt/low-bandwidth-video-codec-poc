# Faces at 42 kbps

A small experiment comparing a neural video codec (DCVC) against standard H.264 at
matched, ultra-low bitrates, to see which one keeps a talking-head video watchable
when bandwidth is very limited. This repo has the code, data, and exact steps to
reproduce it yourself.

No GPU required. Everything below runs on CPU. It's slow (~5.5 sec/frame) but correct.

## Result

At 42 kbps, roughly bad-connection territory, DCVC scored 31.7 dB PSNR against
H.264's 23.6 dB on the same clip at the same file size.

![Akiyo clip: original vs DCVC vs H.264 at 42 kbps](images/akiyo_comparison.png)

The gap holds under real motion too, tested separately on a clip with actual head
turns and camera movement:

![Foreman clip: original vs DCVC vs H.264 at 97 kbps](images/foreman_comparison.png)

Full numbers across four bitrates are in [Results we got](#results-we-got-for-reference)
below, and the raw JSON output from every run is in [`results/`](results).

## 0. Prerequisites

- Python 3.10 to 3.12
- `ffmpeg` with `libx264` support (the [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) Windows builds work fine)
- ~500MB disk space for model checkpoints
- No CUDA/GPU needed for this path (the newer "DCVC-RT" variant does need one, see the note at the bottom)

## 1. Get the code

```bash
git clone --depth 1 https://github.com/microsoft/DCVC.git
cd DCVC/DCVC-family/DCVC
```

Everything else below happens inside this `DCVC/DCVC-family/DCVC` directory unless noted.

## 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Note: `torchvision` is not listed in `requirements.txt` but is imported by the code.
Install it explicitly or you'll hit `ModuleNotFoundError: No module named 'torchvision'`.

## 3. Download the pretrained models

Two separate sets of checkpoints are needed: an "I-frame" (keyframe) image compressor,
and the actual DCVC video model.

**I-frame models** (public, direct download, scripted):
```bash
cd checkpoints
python download_compressai_models.py
cd ..
```
This pulls 8 files (`cheng2020-anchor-*.pth.tar`, `bmshj2018-hyperprior-*.pth.tar`) from
a public S3 bucket, ~380MB total.

**DCVC video models** (manual, this is the one annoying step):
The video-model weights live in a Microsoft OneDrive folder linked from the repo's
README, and OneDrive's share links only render through a real browser (`curl`/`wget`
just get redirected into an unusable SharePoint page, HTTP 403). Open the link in a
browser, select the 4 files named `model_dcvc_quality_0_psnr.pth` through
`model_dcvc_quality_3_psnr.pth` (~37MB each), download, and place them in `checkpoints/`.
There are also `_msssim.pth` variants, not needed for this test, which uses the PSNR
objective throughout.

You should end up with these in `checkpoints/`:
```
cheng2020-anchor-3-e49be189.pth.tar   cheng2020-anchor-4-98b0b468.pth.tar
cheng2020-anchor-5-23852949.pth.tar   cheng2020-anchor-6-4c052b1a.pth.tar
bmshj2018-hyperprior-ms-ssim-*.pth.tar (x4, unused here but downloaded by the script)
model_dcvc_quality_0_psnr.pth ... model_dcvc_quality_3_psnr.pth
```

Quality preset to matching I-frame checkpoint:

| DCVC quality | I-frame checkpoint |
|---|---|
| 0 | `cheng2020-anchor-3-e49be189.pth.tar` |
| 1 | `cheng2020-anchor-4-98b0b468.pth.tar` |
| 2 | `cheng2020-anchor-5-23852949.pth.tar` |
| 3 | `cheng2020-anchor-6-4c052b1a.pth.tar` |

## 4. Get test footage

We used two classic, public-domain CIF (352x288) test sequences from Xiph.org's derf
collection, standard in video-compression research since the 1990s:

```bash
curl -o akiyo_cif.y4m   https://media.xiph.org/video/derf/y4m/akiyo_cif.y4m
curl -o foreman_cif.y4m https://media.xiph.org/video/derf/y4m/foreman_cif.y4m
```

DCVC requires frame dimensions to be a multiple of 64, so crop 352x288 down to
320x256 and extract 16 frames as PNGs (place these *outside* the DCVC repo, e.g. a
sibling `testdata/` folder):

```bash
mkdir -p testdata/akiyo_320x256/akiyo_320x256
ffmpeg -i akiyo_cif.y4m -vframes 16 -vf "crop=320:256:16:16" \
  testdata/akiyo_320x256/akiyo_320x256/im%05d.png

mkdir -p testdata/foreman_320x256/foreman_320x256
ffmpeg -i foreman_cif.y4m -vframes 16 -vf "crop=320:256:16:16" \
  testdata/foreman_320x256/foreman_320x256/im%05d.png
```

## 5. Write the dataset config

DCVC's test script wants a JSON manifest. Save as `poc_dataset_config.json` inside the
`DCVC/DCVC-family/DCVC` directory (adjust `base_path` to wherever your `testdata` folder
actually is, relative to that directory). A copy of the exact configs used here is in
[`results/`](results).

```json
{
    "AkiyoPOC": {
        "base_path": "../../../testdata/akiyo_320x256",
        "sequences": {
            "akiyo_320x256": {"frames": 16, "gop": 16}
        }
    }
}
```

Make a second one, `poc_dataset_config_foreman.json`, pointing at `foreman_320x256`
the same way.

## 6. Run DCVC (once per quality preset)

```bash
python test_video.py \
  --i_frame_model_name cheng2020-anchor \
  --i_frame_model_path checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
  --test_config poc_dataset_config.json \
  --cuda false --worker 1 \
  --output_json_result_path poc_result_q0.json \
  --model_type psnr \
  --recon_bin_path poc_recon_q0 \
  --write_recon_frame true --write_stream false \
  --model_path checkpoints/model_dcvc_quality_0_psnr.pth
```

Repeat for quality 1, 2, 3. Swap the `--i_frame_model_path` and `--model_path` per the
table above, and change the output/recon folder names so they don't overwrite each
other (`poc_result_q1.json`, `poc_recon_q1`, etc.).

`--write_stream false` skips writing an actual compressed bitstream to disk (which
needs a compiled C++ extension you'd otherwise have to build) and instead reports the
*entropy-estimated* bits-per-pixel, the standard way these models are evaluated in
research, and what the numbers below are built from. `--write_recon_frame true` saves
the actual rebuilt PNG frames so you can look at them, at
`poc_recon_q0/akiyo_320x256/model_dcvc_quality_0_psnr/recon_frame_*.png`.

Each run takes roughly 90 seconds on a laptop CPU for 16 frames. Read the resulting
`poc_result_q*.json`. The field you want is `ave_all_frame_bpp` (bits per pixel) and
`ave_all_frame_quality` (PSNR in dB).

To convert bpp to kbps for a given resolution/framerate:
```
kbps = bpp * width * height * fps / 1000
```
(320 x 256 x 29.97fps, in our case.)

Run the same command against `poc_dataset_config_foreman.json` (with `quality 0`) to
get the motion-clip comparison point.

## 7. Build the matched H.264 baseline

For each DCVC result, encode the same PNG frames with H.264 at the same bitrate DCVC
used (take the kbps number from step 6):

```bash
ffmpeg -framerate 29.97 -i testdata/akiyo_320x256/akiyo_320x256/im%05d.png \
  -c:v libx264 -x264-params "nal-hrd=cbr" \
  -b:v 42k -minrate 42k -maxrate 42k -bufsize 10k -g 16 -pix_fmt yuv420p \
  akiyo_h264_42k.mp4
```

Check ffmpeg's own log line `kb/s: ...` to see the actual achieved bitrate. It won't
match your target exactly, use the real one for a fair comparison. Also note this is
the encoder's own bitrate for the picture data only. Don't use `ffprobe`'s file-level
`bit_rate` on a clip this short, fixed container overhead (the MP4 `moov` atom etc.)
dominates the numbers on a sub-second file and will overstate the real bitrate.

Decode it back to PNGs:
```bash
mkdir h264_recon_42k
ffmpeg -i akiyo_h264_42k.mp4 -pix_fmt rgb24 h264_recon_42k/im%05d.png
```

## 8. Measure PSNR (apples-to-apples with DCVC's own metric)

DCVC's own PSNR is computed on RGB pixels, normalized 0 to 1. That is not the same as
ffmpeg's built-in `psnr` filter, which works in YUV per-plane and will give you a
different, non-comparable number. Use [`compute_psnr.py`](compute_psnr.py) instead:

```bash
python compute_psnr.py testdata/akiyo_320x256/akiyo_320x256 h264_recon_42k 16
```

## 9. Build a side-by-side comparison image (optional, for visual sanity-checking)

```python
from PIL import Image
orig = Image.open("testdata/akiyo_320x256/akiyo_320x256/im00016.png").convert("RGB")
dcvc = Image.open("DCVC/DCVC-family/DCVC/poc_recon_q0/akiyo_320x256/model_dcvc_quality_0_psnr/recon_frame_15.png").convert("RGB")
h264 = Image.open("h264_recon_42k/im00016.png").convert("RGB")
w, h = orig.size
combo = Image.new("RGB", (w * 3 + 20, h + 30), "white")
combo.paste(orig, (0, 30))
combo.paste(dcvc, (w + 10, 30))
combo.paste(h264, (2 * w + 20, 30))
combo.save("comparison.png")
```

## Results we got, for reference

| Bitrate | DCVC PSNR | H.264 PSNR |
|---|---|---|
| 42 kbps | 31.7 dB | 23.6 dB |
| 65 kbps | 34.6 dB | 26.8 dB |
| 91 kbps | 36.0 dB | 27.4 dB |
| 136 kbps | 37.0 dB | 27.3 dB |
| 97 kbps (Foreman, motion) | 31.9 dB | 27.4 dB |

Raw JSON output for each of these runs is in [`results/`](results).

## Known gotchas

- If `ffmpeg` complains `Unrecognized option 'vsync'`, drop that flag. Recent ffmpeg
  builds removed it in favor of `-fps_mode`.
- If PSNR numbers look wildly wrong, double check the two PNG sequences you're
  comparing actually have the same framerate assumption. Mismatched `-framerate` on
  input vs. the source file causes ffmpeg to misalign frames when decoding for
  comparison.
- The 16-frame, 0.5-second clip length here is short enough that H.264's own bitrate
  controller may not fully settle, especially at higher target bitrates. Treat the
  H.264 numbers at 91/136 kbps as a bit conservative, not a hard ceiling.
- This tests against plain H.264, which is what WhatsApp actually uses for video
  calls. Other apps use more, e.g. Google Meet runs VP9 with SVC, which is already
  meaningfully more efficient than H.264 on its own. A VP9 baseline isn't included
  here yet, so treat the H.264 numbers as representative of WhatsApp specifically,
  not of every video call app.

## Going further: DCVC-RT (not covered above)

Microsoft's newer, much faster `DCVC-RT` variant (claims 100+ fps on a desktop GPU)
lives in the same repo under `DCVC-family/DCVC-RT`, but its `--write_stream` flag is
mandatory (the script asserts on it), which means it needs a compiled C++ extension
for the actual entropy/bitstream code. You'll need `cmake`, `ninja`, and a C++
compiler (MSVC on Windows, or `build-essential` on Linux) to build it, plus ideally an
NVIDIA/CUDA GPU to actually see its speed advantage. This was skipped for this POC
specifically because none of those were available.

## License

Code in this repo (`compute_psnr.py`, configs) is provided as-is for reproducing the
experiment above. DCVC itself is Microsoft's, under its own license in that repo.
