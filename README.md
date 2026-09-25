# Faces at 42 kbps

A small experiment comparing neural video codecs (DCVC and MLVC) against standard
H.264 at matched, ultra-low bitrates, to see which one keeps a talking-head video
watchable when bandwidth is very limited. This repo has the code, data, and exact
steps to reproduce it yourself.

No GPU required. Everything below runs on CPU. It's slow but correct.

## Result

On a full 10-second clip (300 frames), matched to roughly 46 kbps, bad-connection
territory, at the same file size:

| Codec | Bitrate | PSNR |
|---|---|---|
| H.264 | 45.8 kbps | 29.4 dB |
| DCVC (2021) | 46.6 kbps | 31.7 dB |
| MLVC (2026) | 46.0 kbps | 37.9 dB |

![Akiyo clip: original vs MLVC vs DCVC vs H.264, all at roughly 46 kbps](images/akiyo_comparison_labeled.png)

A single still frame isn't enough to judge a video codec fairly: every
frame here is predicted from the one before it, so quality drifts across a clip in
a way one frame can hide, and a fraction-of-a-second clip isn't long enough to see
that drift happen more than once. Here's the full 10-second clip playing, and
quality plotted frame by frame across all 300 frames instead of eyeballed at one
point in time:

<video src="videos/akiyo_comparison.mp4" controls width="600"></video>

![Per-frame PSNR across the full 300-frame clip](images/akiyo_per_frame_psnr.png)

The chart shows something a single frame or a 16-frame snippet can't: DCVC's
quality visibly saws up and down on a strict 16-frame cycle (its GOP length),
climbing right after each keyframe and drifting back down until the next one, for
all ~19 cycles in the clip. H.264 stays consistently lowest without that same sharp
pattern. MLVC stays highest and comparatively flat the entire time, no repeating
sawtooth. Raw per-frame data for all three codecs is in [`results/`](results)
(`akiyo_mlvc_per_frame_results.json`, `akiyo_dcvc_per_frame.json`,
`akiyo_h264_per_frame.json`).

MLVC is a newer codec from the same research lineage as DCVC (more on what makes
it different [below](#mlvc-a-newer-production-oriented-codec)). Averaged across the
clip it beats DCVC by 6.2 dB at essentially the same bitrate, and both leave H.264
far behind.

The DCVC vs H.264 gap holds under real motion too, tested separately on a clip
with actual head turns and camera movement:

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
320x256. The main result in this repo uses the full clip, 300 frames (Akiyo is
10.01 seconds at 29.97fps); an earlier, shorter pass used just 16 frames (0.5s)
before it became clear that's too short to see quality drift over time, see
[Result](#result) above. Extract as many frames as your source has, up to
`-vframes 300` for the full Akiyo clip (place these *outside* the DCVC repo, e.g. a
sibling `testdata/` folder):

```bash
mkdir -p testdata/akiyo_320x256/akiyo_320x256
ffmpeg -i akiyo_cif.y4m -vframes 300 -vf "crop=320:256:16:16" \
  testdata/akiyo_320x256/akiyo_320x256/im%05d.png

mkdir -p testdata/foreman_320x256/foreman_320x256
ffmpeg -i foreman_cif.y4m -vframes 16 -vf "crop=320:256:16:16" \
  testdata/foreman_320x256/foreman_320x256/im%05d.png
```

## 5. Run DCVC (once per quality preset)

[`scripts/run_dcvc.py`](scripts/run_dcvc.py) wraps DCVC's own `test_video.py`: it
writes the dataset config JSON for you (DCVC's own manifest format, a copy of the
exact configs used here is still in [`results/`](results) if you want to see it) and
maps quality preset to the right checkpoint pair automatically.

```bash
python scripts/run_dcvc.py \
  --dcvc-dir DCVC/DCVC-family/DCVC \
  --frames-dir testdata/akiyo_320x256 \
  --sequence-name akiyo_320x256 \
  --num-frames 300 --gop 16 \
  --quality 0 \
  --out-prefix akiyo_q0
```

Repeat with `--quality 1`, `2`, `3` (using a different `--out-prefix` each time so
outputs don't overwrite each other) to get the full rate-distortion curve. On a
laptop CPU, expect roughly 90 seconds for 16 frames or ~18 minutes for the full
300-frame clip. It's deterministic since there's no randomness in inference, run it
twice and you'll get the exact same bpp/PSNR.

The script prints the resulting PSNR (dB) and bits-per-pixel, and tells you where the
reconstructed PNG frames and full result JSON landed. To convert bpp to kbps for a
given resolution/framerate:
```
kbps = bpp * width * height * fps / 1000
```
(320 x 256 x 29.97fps, in our case.)

Run the same command against a `--frames-dir` pointing at `foreman_320x256` (with
`--quality 0`) to get the motion-clip comparison point.

## 6. Build the matched H.264 baseline

[`scripts/run_h264_baseline.py`](scripts/run_h264_baseline.py) encodes the same PNG
frames with H.264 at a target bitrate, decodes them back, and prints the *actual*
achieved bitrate (take the kbps number from step 5 as your target):

```bash
python scripts/run_h264_baseline.py \
  --frames-glob "testdata/akiyo_320x256/akiyo_320x256/im%05d.png" \
  --fps 29.97 --gop 16 \
  --target-kbps 42 --bufsize-kbps 10 \
  --out-video akiyo_h264_42k.mp4 \
  --recon-dir h264_recon_42k
```

Use the actual achieved bitrate it prints, not your `--target-kbps`, when comparing
PSNR at "matched bitrate", x264's own rate control won't hit the target exactly,
especially on a clip this short. Note this is the encoder's own bitrate for the
picture data only. Don't separately check `ffprobe`'s file-level `bit_rate` on a clip
this short, fixed container overhead (the MP4 `moov` atom etc.) dominates the numbers
on a sub-second file and will overstate the real bitrate.

One more short-clip quirk: x264's multithreaded rate control isn't perfectly
deterministic, re-running the exact same command can land a percent or two off from
a previous run (we saw 41.06 kbps and 40.11 kbps across two runs of the identical
42 kbps target). Not a bug, just don't expect bit-identical output run to run.

## 7. Measure PSNR (apples-to-apples with DCVC's own metric)

DCVC's own PSNR is computed on RGB pixels, normalized 0 to 1. That is not the same as
ffmpeg's built-in `psnr` filter, which works in YUV per-plane and will give you a
different, non-comparable number. Use [`scripts/compute_psnr.py`](scripts/compute_psnr.py)
for a quick average, or [`scripts/compute_per_frame_psnr.py`](scripts/compute_per_frame_psnr.py)
if you want the full per-frame breakdown (needed to see quality drift across a clip,
see [Result](#result) above for why that matters):

```bash
# quick average only
python scripts/compute_psnr.py testdata/akiyo_320x256/akiyo_320x256 h264_recon_42k 16

# full per-frame data as JSON (handles DCVC's unpadded 0-indexed recon_frame_N.png
# naming vs ffmpeg's zero-padded 1-indexed imNNNNN.png naming via --orig-start/--recon-start)
python scripts/compute_per_frame_psnr.py \
  --orig-dir testdata/akiyo_320x256/akiyo_320x256 --orig-pattern "im{i:05d}.png" --orig-start 1 \
  --recon-dir h264_recon_42k --recon-pattern "im{i:05d}.png" --recon-start 1 \
  --num-frames 16 --out h264_per_frame.json
```

## 8. Build a side-by-side comparison image and video

[`scripts/build_comparison_image.py`](scripts/build_comparison_image.py) composes any
number of PNGs side by side with labels:

```bash
python scripts/build_comparison_image.py \
  --panel "Original" "" testdata/akiyo_320x256/akiyo_320x256/im00016.png \
  --panel "DCVC" "31.7 dB, 42 kbps" DCVC/DCVC-family/DCVC/poc_recon_q0/akiyo_320x256/model_dcvc_quality_0_psnr/recon_frame_15.png \
  --panel "H.264" "23.6 dB, 41 kbps" h264_recon_42k/im00016.png \
  --out comparison.png
```

For an actual playable comparison video (recommended, a still frame can't show the
quality drift covered in [Result](#result)), turn each codec's full frame sequence
into its own short video, label each with `drawtext`, then stack them side by side:

```bash
# one short mp4 per source (repeat for each codec's own recon frame sequence)
ffmpeg -framerate 29.97 -i testdata/akiyo_320x256/akiyo_320x256/im%05d.png \
  -pix_fmt yuv420p -c:v libx264 -crf 15 orig.mp4

# then, with a TTF font file (e.g. arial.ttf) copied next to your working directory
# to sidestep ffmpeg treating a Windows drive-letter colon as a filter separator:
ffmpeg -i orig.mp4 -i dcvc.mp4 -i h264.mp4 -filter_complex "
[0:v]drawtext=fontfile=arial.ttf:text='Original':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[v0];
[1:v]drawtext=fontfile=arial.ttf:text='DCVC (31.7dB, 42kbps)':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[v1];
[2:v]drawtext=fontfile=arial.ttf:text='H.264 (23.6dB, 41kbps)':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[v2];
[v0][v1][v2]hstack=inputs=3
" -c:v libx264 -crf 18 -pix_fmt yuv420p side_by_side.mp4
```

## Results we got, for reference

Full 300-frame clip, all three codecs, matched bitrate (the headline result above):

| Codec | Bitrate | PSNR |
|---|---|---|
| H.264 | 45.8 kbps | 29.4 dB |
| DCVC | 46.6 kbps | 31.7 dB |
| MLVC | 46.0 kbps | 37.9 dB |

DCVC's rate-distortion curve (DCVC vs H.264 only), from an earlier pass on a shorter
16-frame snippet, before the full-clip run above replaced it as the headline result:

| Bitrate | DCVC PSNR | H.264 PSNR |
|---|---|---|
| 42 kbps | 31.7 dB | 23.6 dB |
| 65 kbps | 34.6 dB | 26.8 dB |
| 91 kbps | 36.0 dB | 27.4 dB |
| 136 kbps | 37.0 dB | 27.3 dB |
| 97 kbps (Foreman, motion, 16 frames) | 31.9 dB | 27.4 dB |

Raw JSON output for each of these runs is in [`results/`](results).

## Known gotchas

- If `ffmpeg` complains `Unrecognized option 'vsync'`, drop that flag. Recent ffmpeg
  builds removed it in favor of `-fps_mode`.
- If PSNR numbers look wildly wrong, double check the two PNG sequences you're
  comparing have the same framerate assumption. Mismatched `-framerate` on
  input vs. the source file causes ffmpeg to misalign frames when decoding for
  comparison.
- The 16-frame, 0.5-second clip length here is short enough that H.264's own bitrate
  controller may not fully settle, especially at higher target bitrates. Treat the
  H.264 numbers at 91/136 kbps as a bit conservative, not a hard ceiling.
- This tests against plain H.264, which is what WhatsApp uses for video
  calls. Other apps use more, e.g. Google Meet runs VP9 with SVC, which is already
  meaningfully more efficient than H.264 on its own. A VP9 baseline isn't included
  here yet, so treat the H.264 numbers as representative of WhatsApp specifically,
  not of every video call app.

## MLVC: a newer, production-oriented codec

[MLVC](https://github.com/microsoft/mlvc) is a newer neural video codec Microsoft
released in 2026, built to run on real hardware (phone NPUs, CPUs) with a
production entropy coder, rather than a research-only setup like DCVC's.

```bash
git clone https://github.com/microsoft/mlvc.git
cd mlvc
uv sync --extra onnxruntime   # CPU backend
```

To produce real bitstreams (real compressed bytes, not an estimate) you need its C++ entropy
coder built, which needs a C++ compiler:

```bash
# on Windows, from a shell with vcvars64.bat sourced (MSVC installed via
# Visual Studio Build Tools, "Desktop development with C++" workload):
uv pip install packages/msrtc_rans
```

Download a checkpoint (MLVC or the smaller MLVC-S, both PSNR or perceptual
objective) from the URLs in the [MLVC README](https://github.com/microsoft/mlvc#models),
and verify its SHA-256 against the hash listed there.

[`scripts/run_mlvc.py`](scripts/run_mlvc.py) is the actual script used to produce
every MLVC number in this repo, adapted from MLVC's own
[`video/notebooks/demo.ipynb`](https://github.com/microsoft/mlvc/blob/main/video/notebooks/demo.ipynb)
into a plain CLI. Copy it into the mlvc repo's `video/` directory (it imports MLVC's
own `src` package, so it needs to live there or have that directory on `PYTHONPATH`),
convert your clip to raw YUV420 first (`ffmpeg -i clip.mp4 -pix_fmt yuv420p -f rawvideo clip.yuv`),
then run:

```bash
uv run python run_mlvc.py \
  --checkpoint /path/to/mlvc-psnr-v1.ckpt \
  --video /path/to/akiyo_320x256_long.yuv \
  --width 320 --height 256 --fps 29.97 \
  --q-index 50 \
  --out-dir ./out --save-frames
```

`--q-index` ranges 0 to 63 (higher = more bitrate/quality). The right value to hit a
given target bitrate depends heavily on your content and clip length. It's not a
fixed lookup: on the 16-frame snippet used during early testing, `q_index=21` landed
at ~38 kbps, but on the full 300-frame clip (long, mostly-static content compresses
much better on average) that same `q_index=21` dropped to just 9.2 kbps. `q_index=50`
is what landed at ~46 kbps on the full clip. Try a few values and check the
`kbps` it reports rather than assuming a number that worked on a different clip.

On the full 300-frame Akiyo clip, at a closely matched ~46 kbps: MLVC scored 37.9 dB,
beating DCVC's 31.7 dB by 6.2 dB while using slightly less bitrate. Full
per-quality-preset results (on the earlier short snippet) are in
[`results/akiyo_mlvc_results.json`](results/akiyo_mlvc_results.json) and full-clip
per-frame data in [`results/akiyo_mlvc_per_frame_results.json`](results/akiyo_mlvc_per_frame_results.json).

**Gotcha:** on Windows, `scipy` (an MLVC dependency) failed to import with
`DLL load failed... An Application Control policy has blocked this file`. That's
Windows 11's Smart App Control, which blocks unsigned or unrecognized binaries,
including normal PyPI wheels with compiled extensions. If you hit this, check
`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` for
`VerifiedAndReputablePolicyState`. Once Smart App Control is in Enforce mode (`1`),
Microsoft only supports turning it off via a full Windows reset, there's no
per-app exception. Not a bug in MLVC, just what a locked-down Windows machine does
to ML tooling.

## Going further: DCVC-RT (not covered above)

Microsoft's newer, much faster `DCVC-RT` variant (claims 100+ fps on a desktop GPU)
lives in the same repo under `DCVC-family/DCVC-RT`, but its `--write_stream` flag is
mandatory (the script asserts on it), which means it needs a compiled C++ extension
for the actual entropy/bitstream code. You'll need `cmake`, `ninja`, and a C++
compiler (MSVC on Windows, or `build-essential` on Linux) to build it, plus ideally an
NVIDIA/CUDA GPU to see its speed advantage. This was skipped for this POC
specifically because none of those were available.

## License

Code in this repo (`scripts/`, configs) is provided as-is for reproducing the
experiment above. DCVC and MLVC are Microsoft's, under their own licenses in
their own repos.
