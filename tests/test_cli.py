import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from qr_video import cli as cli_module
from qr_video.profiles import Profile
from qr_video.transport import SHARD_SIZE

VIDEO_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def cli(
    *args: str | Path,
    success: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "qr_video", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, result.stdout
        assert "Traceback" not in result.stderr
    return result


@pytest.mark.parametrize("command", ["keygen", "encode", "decode", "stats"])
def test_help(command: str) -> None:
    assert "usage:" in cli(command, "--help").stdout


def test_installed_console_entry_point() -> None:
    executable = Path(sys.executable).parent / "qr-video"
    result = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "qr-video 0.1.0"


def test_keygen_and_stats(tmp_path: Path) -> None:
    key = tmp_path / "secret.key"
    cli("keygen", "--out", key)
    assert len(key.read_bytes()) == 32
    assert key.read_text(encoding="ascii").isalnum()
    original_key = key.read_bytes()
    cli("keygen", "--out", key, success=False)
    assert key.read_bytes() == original_key
    source = tmp_path / "input.bin"
    source.write_bytes(b"\x00\xff" * 1024)
    stats = json.loads(
        cli("stats", source, "--key", key.read_text(encoding="ascii"), "--compress").stdout
    )
    assert stats["encrypted"] is True
    assert stats["compressed"] is True
    assert stats["original_bytes"] == 2048
    assert stats["shard_payload_bytes"] == 342
    assert stats["packet_overhead_bytes"] == 40
    assert stats["stream_bytes"] < stats["original_bytes"]
    assert stats["application_mb_per_hour"] == pytest.approx(
        2048 / stats["duration_seconds"] * 3600 / 1_000_000
    )
    assert stats["video_frames"] == stats["images"] * 3


@pytest.mark.parametrize("command", ["encode", "stats", "decode"])
@pytest.mark.parametrize("key", ["abhi2810", "spaces and punctuation!", "clé🔑", "x" * 100])
def test_inline_key_parsing(command: str, key: str, tmp_path: Path) -> None:
    arguments = [command, str(tmp_path / "input"), "--key", key]
    if command != "stats":
        arguments.extend(["--out", str(tmp_path / "output")])
    assert cli_module.parser().parse_args(arguments).key == key.encode("utf-8")


def test_empty_inline_key_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"data")
    result = cli("stats", source, "--key", "", success=False)
    assert "key must not be empty" in result.stderr


def test_progress_renders_percent_and_frame_count() -> None:
    stream = TTYBuffer()
    encoding = cli_module._Progress("Encoding", "QR images", total=4, stream=stream)
    encoding.update(2)
    encoding.update(4)
    encoding.close()
    decoding = cli_module._Progress("Decoding", "video frames", stream=stream)
    decoding.update(7)
    decoding.close()
    output = stream.getvalue()
    assert "Encoding:  50% (2/4 QR images)" in output
    assert "Encoding: 100% (4/4 QR images)" in output
    assert "Decoding: \\ 7 video frames" in output


def test_progress_is_silent_when_stderr_is_not_a_tty() -> None:
    stream = io.StringIO()
    progress = cli_module._Progress("Encoding", "QR images", total=1, stream=stream)
    progress.update(1)
    progress.close()
    assert stream.getvalue() == ""


def test_inline_key_and_key_file_are_mutually_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"data")
    key = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
    result = cli("stats", source, "--key", key, "--key-file", source, success=False)
    assert "not allowed with argument" in result.stderr


@pytest.mark.parametrize(
    "options",
    [
        ["--repeat", "0"],
        ["--fps", "30", "--repeat", "7"],
        ["--data-shards", "0"],
        ["--parity-shards", "101"],
        ["--interleave", "17"],
        ["--crf", "52"],
        ["--crf", "-1"],
    ],
)
def test_invalid_encode_options(tmp_path: Path, options: list[str]) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"test")
    output = tmp_path / "invalid.mp4"
    cli("encode", source, "--out", output, *options, success=False)
    assert not output.exists()


def test_path_and_key_errors(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"keep source")
    cli("encode", source, "--out", source, "--overwrite", success=False)
    assert source.read_bytes() == b"keep source"
    alias = tmp_path / "alias"
    os.link(source, alias)
    cli("encode", source, "--out", alias, "--overwrite", success=False)
    assert source.read_bytes() == b"keep source"
    key = tmp_path / "secret.key"
    cli("keygen", "--out", key)
    key_before = key.read_bytes()
    cli("encode", source, "--out", key, "--key-file", key, "--overwrite", success=False)
    cli("encode", key, "--out", tmp_path / "key.mp4", "--key-file", key, success=False)
    assert key.read_bytes() == key_before
    bad_key = tmp_path / "bad.key"
    bad_key.write_bytes(b"")
    output = tmp_path / "invalid.mp4"
    result = cli("encode", source, "--out", output, "--key-file", bad_key, success=False)
    assert "must not be empty" in result.stderr
    cli("encode", source, "--out", output, "--key-file", tmp_path / "missing.key", success=False)
    assert not output.exists()
    output.write_bytes(b"keep output")
    cli("encode", source, "--out", output, success=False)
    assert output.read_bytes() == b"keep output"


def test_missing_ffmpeg_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"test")
    output = tmp_path / "output.mp4"
    env = dict(os.environ, PATH=str(tmp_path / "no-tools"))
    result = cli("encode", source, "--out", output, success=False, env=env)
    assert "ffmpeg" in result.stderr.lower()
    assert not output.exists()


def test_output_directory_alias_is_frozen_before_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"keep input")
    first = tmp_path / "first"
    first.mkdir()
    alias = tmp_path / "output"
    alias.symlink_to(first, target_is_directory=True)

    def fake_video(
        packets: Iterable[bytes],
        destination: Path,
        *,
        profile: Profile,
        crf: int,
        on_progress: object,
    ) -> int:
        count = sum(1 for _ in packets)
        destination.write_bytes(b"complete video fixture")
        alias.unlink()
        alias.symlink_to(tmp_path, target_is_directory=True)
        return (count + 1) // 2

    monkeypatch.setattr(cli_module, "write_video", fake_video)
    assert (
        cli_module.main(
            ["encode", str(source), "--out", str(alias / "input.bin"), "--overwrite"],
        )
        == 0
    )
    assert source.read_bytes() == b"keep input"
    assert (first / "input.bin").read_bytes() == b"complete video fixture"


def test_unrecoverable_video(tmp_path: Path) -> None:
    source = tmp_path / "not-video.mp4"
    source.write_bytes(b"not a video")
    output = tmp_path / "failed"
    cli("decode", source, "--out", output, success=False)
    assert not output.exists()
    for limit in ("0", "-1", str(64 * 1024**3 + 1)):
        cli("decode", source, "--out", output, "--max-bytes", limit, success=False)
        assert not output.exists()


@pytest.mark.video
@pytest.mark.skipif(not VIDEO_AVAILABLE, reason="FFmpeg and ffprobe required")
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize(
    "kind", ["empty", "binary", "compressible", "incompressible", "multiblock"]
)
def test_full_cli_roundtrip(tmp_path: Path, kind: str, encrypted: bool) -> None:
    if kind == "empty":
        original = b""
    elif kind == "binary":
        original = bytes(range(256)) * 2
    elif kind == "compressible":
        original = b"\x00\xffcompress me\n" * 6000
    elif kind == "incompressible":
        original = os.urandom(4097)
    else:
        original = os.urandom(SHARD_SIZE * 100 * 2 + 137)
    source = tmp_path / "input.bin"
    source.write_bytes(original)
    output = tmp_path / "encoded.mp4"
    recovered = tmp_path / "recovered.bin"
    key_args: list[str | Path] = []
    if encrypted:
        key = tmp_path / "secret.key"
        cli("keygen", "--out", key)
        key_args = ["--key", key.read_text(encoding="ascii")]
    compression = ["--compress"] if kind == "compressible" else []
    encoded = cli("encode", source, "--out", output, *key_args, *compression)
    stats = json.loads(encoded.stderr.splitlines()[0])
    assert output.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    video_stream = json.loads(probe.stdout)["streams"][0]
    assert (video_stream["width"], video_stream["height"]) == (640, 360)
    assert video_stream["pix_fmt"] == "yuv420p"
    assert video_stream["r_frame_rate"] == "30/1"
    assert int(video_stream["nb_read_frames"]) == stats["video_frames"]
    if kind == "multiblock":
        assert stats["blocks"] >= 3
    cli("decode", output, "--out", recovered, *key_args)
    assert recovered.read_bytes() == original
    assert source.read_bytes() == original


@pytest.mark.video
@pytest.mark.skipif(not VIDEO_AVAILABLE, reason="FFmpeg and ffprobe required")
@pytest.mark.parametrize("within_budget", [False, True])
def test_encrypted_transcode_with_dropped_and_duplicated_frames(
    tmp_path: Path,
    within_budget: bool,
) -> None:
    # Dropping two shards exceeds the tiny last block's budget, but not an 11+3 block's.
    size = SHARD_SIZE * 110 if within_budget else SHARD_SIZE * 100 + 157
    original = os.urandom(size)
    source = tmp_path / "input.bin"
    source.write_bytes(original)
    key = tmp_path / "secret.key"
    cli("keygen", "--out", key)
    video = tmp_path / "encoded.mp4"
    cli("encode", source, "--out", video, "--key-file", key)
    transformed = tmp_path / "transcoded.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(video),
            "-vf",
            "select='gte(n,9)',setpts=N/(30*TB),fps=60",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            str(transformed),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    recovered = tmp_path / "recovered.bin"
    if not within_budget:
        result = cli("decode", transformed, "--out", recovered, "--key-file", key, success=False)
        assert "Insufficient recovery packets for block 1" in result.stderr
        assert not recovered.exists()
        return
    cli("decode", transformed, "--out", recovered, "--key-file", key)
    assert recovered.read_bytes() == original
    wrong_key = tmp_path / "wrong.key"
    cli("keygen", "--out", wrong_key)
    bad_output = tmp_path / "failed"
    cli("decode", video, "--out", bad_output, "--key-file", wrong_key, success=False)
    assert not bad_output.exists()
    cli("decode", video, "--out", bad_output, success=False)
    assert not bad_output.exists()
    bad_output.write_bytes(b"keep existing output")
    cli(
        "decode",
        video,
        "--out",
        bad_output,
        "--key-file",
        wrong_key,
        "--overwrite",
        success=False,
    )
    assert bad_output.read_bytes() == b"keep existing output"
    limited = cli(
        "decode",
        video,
        "--out",
        bad_output,
        "--key-file",
        key,
        "--max-bytes",
        "10",
        "--overwrite",
        success=False,
    )
    assert "limit" in limited.stderr.lower()
    assert bad_output.read_bytes() == b"keep existing output"
    cli("decode", video, "--out", bad_output, "--key-file", key, "--overwrite")
    assert bad_output.read_bytes() == original


@pytest.mark.video
@pytest.mark.skipif(not VIDEO_AVAILABLE, reason="FFmpeg and ffprobe required")
def test_custom_timing_and_coding(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    original = os.urandom(4096)
    source.write_bytes(original)
    video = tmp_path / "encoded.mp4"
    result = cli(
        "encode",
        source,
        "--out",
        video,
        "--fps",
        "24",
        "--repeat",
        "2",
        "--data-shards",
        "4",
        "--parity-shards",
        "1",
        "--interleave",
        "2",
    )
    stats = json.loads(result.stderr.splitlines()[0])
    assert stats["fps"] == 24
    assert stats["repeat"] == 2
    assert stats["images_per_second"] == 12
    recovered = tmp_path / "recovered"
    cli("decode", video, "--out", recovered)
    assert recovered.read_bytes() == original
    key = tmp_path / "secret.key"
    cli("keygen", "--out", key)
    cli("decode", video, "--out", tmp_path / "downgrade", "--key-file", key, success=False)
    assert not (tmp_path / "downgrade").exists()
