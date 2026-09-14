# Project guide

## Code

- Support Python 3.11 and newer. Source lives in `src/qr_video`; tests live in `tests`.
- Keep file and video processing streaming. Do not load an entire transfer into memory.
- Treat envelope and packet layouts as versioned protocols. Preserve compatibility unless a format change is intentional and tested.
- Preserve safe output behavior: no partial files, no silent overwrite, and no publication before integrity and authentication checks pass.
- Never add real keys, generated videos, recovered files, build output, or virtual environments to Git.

## Checks

Install development dependencies with `python -m pip install -e '.[dev]'`.

Run these before finishing:

```sh
pytest -q
ruff check .
mypy src/qr_video
python -m build
```

Tests marked `video` require `ffmpeg` and `ffprobe` on `PATH`. During quick iterations, use `pytest -q -m "not video"`, but run the full suite for changes to QR rendering, video handling, transport recovery, or the CLI.

Add or update tests for behavior changes. Prefer focused tests near the affected module, including failure cases that verify no output is published.
