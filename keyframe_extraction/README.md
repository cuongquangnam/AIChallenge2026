# Key Frame Extraction

## Project overview

This standalone offline-indexing module processes one video at a time. The
official pretrained TransNet V2 model detects shot boundaries, and the module
extracts exactly three representative JPEG files from every shot: its start,
floor-based middle, and end frames. It also writes complete shot, video, timing,
and filename metadata to `shots.json`.

This project implements only shot detection and keyframe extraction. It is not a
complete multimodal retrieval system.

## Architecture

```text
Video
  ↓
TransNet V2 shot detection
  ↓
Inclusive shot boundaries
  ↓
Start / Middle / End frame selection
  ↓
JPEG keyframes + shots.json
```

TransNet V2 performs inference once for the input video. OpenCV then extracts the
selected full-resolution frames. Metadata is published atomically only after all
keyframes have been decoded and encoded successfully.

## Requirements

- Python 3.9, 3.10, or 3.11 (64-bit)
- Git and Git LFS, so pip can install the pinned TransNet V2 source and weights
- A system-level `ffmpeg` executable available on `PATH`
- Enough memory for TransNet V2's low-resolution inference frames and the
  requested full-resolution keyframes

The pinned TensorFlow build runs on CPU without additional configuration.
TensorFlow may use a supported GPU when the platform-specific CUDA environment
is correctly installed. Native Windows TensorFlow releases after 2.10 do not
provide GPU support; use WSL2 for current NVIDIA GPU support. The standard macOS
TensorFlow package runs this project on CPU; platform-specific acceleration and
its compatibility depend on the TensorFlow/Apple stack.

The end-to-end macOS validation for this repository used Python 3.9.6 on Apple
Silicon (`arm64`), TensorFlow 2.15.1, and FFmpeg 8.1.2. All pinned packages in
`requirements.txt` provided compatible native arm64 wheels for that environment.

Install system dependencies on macOS with Homebrew:

```bash
brew install ffmpeg git git-lfs
git lfs install
```

Install them on Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg git git-lfs python3-venv
git lfs install
```

The official TransNet V2 package does not declare its inference dependencies.
This project's `requirements.txt` therefore pins TensorFlow, NumPy,
`ffmpeg-python`, Pillow, and OpenCV explicitly. Package installation requires
network access, but normal video processing makes no network requests.

## Setup

From the directory containing this project:

```bash
cd keyframe_extraction
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

On Windows Command Prompt, activate the environment with:

```bat
.venv\Scripts\activate.bat
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

## Usage

Basic invocation:

```bash
python3 main.py --input sample.mp4 --output output
```

Use a custom transition threshold:

```bash
python3 main.py --input sample.mp4 --output output --threshold 0.65
```

Set JPEG quality:

```bash
python3 main.py --input sample.mp4 --output output --jpeg-quality 90
```

Allow generated files to be replaced:

```bash
python3 main.py --input sample.mp4 --output output --overwrite
```

Enable detailed logging and failure stack traces:

```bash
python3 main.py --input sample.mp4 --output output --log-level DEBUG
```

Run `python3 main.py --help` to see all CLI options. Displaying help does not
initialize TensorFlow or the TransNet V2 model.

## Deterministic sample and end-to-end verification

The checked-in test workflow uses a locally generated nine-second H.264 video at
25 FPS. It contains three solid red, green, and blue sections separated by hard
cuts and requires no external media. Generate or replace it with:

```bash
mkdir -p test_data
ffmpeg -y \
  -f lavfi -i color=c=red:s=640x360:r=25:d=3 \
  -f lavfi -i color=c=green:s=640x360:r=25:d=3 \
  -f lavfi -i color=c=blue:s=640x360:r=25:d=3 \
  -filter_complex '[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]' \
  -map '[v]' -c:v libx264 -preset medium -crf 18 \
  -g 25 -keyint_min 25 -sc_threshold 0 -movflags +faststart \
  test_data/sample.mp4
```

With `.venv` activated, run the full extraction:

```bash
python3 main.py \
  --input test_data/sample.mp4 \
  --output test_output \
  --overwrite \
  --log-level DEBUG
```

Then verify metadata, boundaries, timestamps, JPEG readability, exact file
counts, temporary-file cleanup, and OpenCV resource release:

```bash
python3 scripts/verify_output.py \
  --video test_data/sample.mp4 \
  --output test_output
```

## Output structure

For a two-shot video, the output is:

```text
output/
├── shot_001_start.jpg
├── shot_001_middle.jpg
├── shot_001_end.jpg
├── shot_002_start.jpg
├── shot_002_middle.jpg
├── shot_002_end.jpg
└── shots.json
```

`shots.json` contains absolute source-video identity, video properties, detector
configuration, inclusive frame boundaries, frame-derived timestamps, and the
three keyframe filenames for every shot.

## Frame-index semantics

- Video frame indices are zero-based.
- Shot IDs are one-based and sequential.
- Shot start and end boundaries are both inclusive, matching the official
  TransNet V2 inference API.
- The middle frame is calculated with `(start + end) // 2`.
- Timestamps are calculated with `frame_index / fps`.
- A one-frame or very short shot can select the same source frame more than once.
  The module still writes all three required, separately named JPEG files.

## Error handling

The CLI exits with a nonzero status and a concise error when:

- the input file does not exist or OpenCV cannot open it;
- the stream is empty or reports invalid FPS, dimensions, or frame count;
- the TransNet V2 package or one of its dependencies is missing;
- the system-level FFmpeg executable is not installed or cannot decode the file;
- TensorFlow or the packaged pretrained weights cannot initialize;
- TransNet V2 returns malformed predictions or shot boundaries;
- an exact target frame cannot be decoded;
- `cv2.imwrite` cannot encode a JPEG or metadata cannot be serialized; or
- a generated output path already exists and `--overwrite` was not supplied.

At `DEBUG` log level, failures include their full stack trace. The output
directory is created recursively. JPEGs are staged first, and `shots.json` is
written through a flushed temporary file in the same directory and replaced
atomically only after all JPEG work succeeds.

## Limitations

Exact random frame seeking depends on the codec, container, and OpenCV video
backend. The extractor verifies the decoder's reported position before accepting
a random-seek result. If that verification fails, it reopens the video and
decodes sequentially from frame zero to the exact requested zero-based index.
Severely corrupted streams can still make a requested frame unavailable; such a
frame is reported as an error and is never silently replaced by a neighbor.

OpenCV and FFmpeg can occasionally report different frame counts for unusual or
damaged containers. In that case, the detector logs a warning and constrains
boundaries to the smaller decodable range.

## Attribution

Shot detection uses [TransNet V2](https://github.com/soCzech/TransNetV2) by
Tomáš Souček and Jakub Lokoč:

> “TransNet V2: An effective deep network architecture for fast shot transition
> detection”

The official TensorFlow/Keras inference package, pretrained weights, and scene
conversion API are installed directly from the upstream repository at the pinned
commit recorded in `requirements.txt`.
