import json
import random
import shutil
import subprocess
import sys
from collections.abc import Generator, Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import zxingcpp
from qrcode.util import MODE_8BIT_BYTE, QRData

from qr_video import video
from qr_video.errors import QRVideoError
from qr_video.profiles import CONSERVATIVE, Profile
from qr_video.video import read_video, render_frame, write_video


def _packets(count: int) -> list[bytes]:
    rng = random.Random(84231)
    return [b"\x00\xff\x80" + rng.randbytes(379) for _ in range(count)]


def _fake_codec(monkeypatch: pytest.MonkeyPatch, script: str) -> list[subprocess.Popen[bytes]]:
    children: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def spawn(arguments: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        child: subprocess.Popen[bytes] = real_popen([sys.executable, "-c", script], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)
    monkeypatch.setattr(video, "_probe_video", lambda source: (640, 360))
    monkeypatch.setattr(video.subprocess, "Popen", spawn)
    return children


def test_render_preserves_raw_binary_packets() -> None:
    packets = [bytes(range(256)) + bytes(range(126)), _packets(1)[0]]
    frame = render_frame(packets[0], packets[1])
    assert frame.shape == (360, 640)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    for packet, (x, y) in zip(packets, CONSERVATIVE.positions, strict=True):
        barcode = zxingcpp.read_barcode(
            frame[y : y + 315, x : x + 315],
            formats=zxingcpp.BarcodeFormats(zxingcpp.BarcodeFormat.QRCode),
        )
        assert barcode is not None
        assert isinstance(barcode.bytes, bytes)
        assert barcode.bytes == packet


@pytest.mark.parametrize(
    "packet", [b"A" * 382, b"1" * 382, b"\x00" * 382], ids=["letters", "digits", "nul"]
)
def test_capacity_accepts_382_bytes_in_forced_byte_mode(
    packet: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes: list[int] = []
    original = video.qrcode.QRCode.add_data

    def add_data(qr: Any, data: QRData, optimize: int = 20) -> None:
        modes.append(data.mode)
        assert data.data == packet
        assert optimize == 0
        original(qr, data, optimize=optimize)

    monkeypatch.setattr(video.qrcode.QRCode, "add_data", add_data)
    frame = render_frame(packet, None)
    assert modes == [MODE_8BIT_BYTE]
    assert list(video._decode_frame(frame, CONSERVATIVE)) == [packet]


@pytest.mark.parametrize(
    "packet", [b"A" * 383, b"1" * 383, b"\x00" * 383], ids=["letters", "digits", "nul"]
)
def test_capacity_rejects_383_bytes_without_text_optimization(packet: bytes) -> None:
    with pytest.raises(QRVideoError, match="383 bytes; maximum is 382"):
        render_frame(packet, None)


@pytest.mark.parametrize("length", [0, 1, 15, 16, 100, 381, 382])
def test_zero_safe_codewords_match_the_qrcode_encoder(length: int) -> None:
    packet = random.Random(317).randbytes(length)
    expected = video.qrcode.util.create_data(
        CONSERVATIVE.version,
        video.qrcode.constants.ERROR_CORRECT_H,
        [QRData(packet, mode=MODE_8BIT_BYTE)],
    )
    assert video._zero_safe_codewords(packet, CONSERVATIVE) == expected


@pytest.mark.parametrize("length", [16, 30, 100, 381, 382])
def test_all_zero_payloads_decode_at_different_lengths(length: int) -> None:
    packet = bytes(length)
    assert list(video._decode_frame(render_frame(packet, None), CONSERVATIVE)) == [packet]


def test_each_qr_has_integer_modules_and_its_own_quiet_zone() -> None:
    left, right = _packets(2)
    frame = render_frame(left, right)
    assert CONSERVATIVE.positions == ((3, 22), (322, 22))
    outside = np.ones(frame.shape, dtype=bool)
    for x, y in CONSERVATIVE.positions:
        qr = frame[y : y + 315, x : x + 315]
        outside[y : y + 315, x : x + 315] = False
        assert np.all(qr[:12] == 255)
        assert np.all(qr[-12:] == 255)
        assert np.all(qr[:, :12] == 255)
        assert np.all(qr[:, -12:] == 255)
        assert np.any(qr[12:-12, 12:-12] == 0)
        assert np.all(qr.reshape(105, 3, 105, 3) == qr[::3, ::3][:, None, :, None])
    assert np.all(frame[outside] == 255)
    assert set(np.unique(frame)) == {0, 255}


def test_odd_packet_has_a_blank_right_panel() -> None:
    packet = _packets(1)[0]
    frame = render_frame(packet, None)
    assert np.all(frame[:, 322:] == 255)
    assert list(video._decode_frame(frame, CONSERVATIVE)) == [packet]


@pytest.mark.parametrize("packet", ["hello", bytearray(b"hello"), 42, None])
def test_render_rejects_nonbyte_packets(packet: Any) -> None:
    with pytest.raises(QRVideoError, match="bytes"):
        render_frame(packet, None)


@pytest.mark.parametrize(
    "profile",
    [
        replace(CONSERVATIVE, version=19),
        replace(CONSERVATIVE, width=800),
        replace(CONSERVATIVE, height=480),
        replace(CONSERVATIVE, pixels_per_module=2),
        replace(CONSERVATIVE, qr_capacity=383),
        replace(CONSERVATIVE, fps=0),
        replace(CONSERVATIVE, repeat=0),
        replace(CONSERVATIVE, fps=121),
        replace(CONSERVATIVE, repeat=121),
        replace(CONSERVATIVE, fps=30, repeat=7),
        replace(CONSERVATIVE, fps=True),
    ],
)
def test_render_rejects_invalid_profiles(profile: Profile) -> None:
    with pytest.raises(QRVideoError):
        render_frame(b"packet", None, profile)


def test_render_checks_qr_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    qr = MagicMock()
    qr.make_image.return_value.convert.return_value = np.ones((300, 300), dtype=np.uint8)
    monkeypatch.setattr(video.qrcode, "QRCode", lambda **kwargs: qr)
    with pytest.raises(QRVideoError, match="dimensions"):
        render_frame(b"packet", None)


def test_decoder_reads_the_other_qr_when_one_is_an_erasure() -> None:
    left, right = _packets(2)
    frame = render_frame(left, right)
    frame[:, :322] = 255
    assert list(video._decode_frame(frame, CONSERVATIVE)) == [right]
    assert list(video._decode_frame(np.full_like(frame, 255), CONSERVATIVE)) == []


def test_decoder_has_a_full_frame_fallback_for_rotated_content() -> None:
    packets = _packets(2)
    frame = np.rot90(render_frame(packets[0], packets[1])).copy()
    assert set(video._decode_frame(frame, CONSERVATIVE)) == set(packets)


def test_decoder_uses_bytes_and_limits_detection_to_qr(monkeypatch: pytest.MonkeyPatch) -> None:
    class Barcode:
        bytes = b"\x00\xff\x80"

        @property
        def text(self) -> str:
            raise AssertionError("Text transcoding would corrupt a binary packet")

    def read_barcodes(image: Any, **kwargs: Any) -> list[Barcode]:
        assert list(kwargs["formats"]) == [zxingcpp.BarcodeFormat.QRCode]
        return [Barcode()]

    monkeypatch.setattr(video.zxingcpp, "read_barcodes", read_barcodes)
    assert list(video._read_qrs(np.zeros((2, 2), dtype=np.uint8))) == [b"\x00\xff\x80"]


@pytest.mark.parametrize("crf", [-1, 52, 18.0, True, "18"])
def test_write_rejects_invalid_crf(crf: Any, tmp_path: Path) -> None:
    with pytest.raises(QRVideoError, match="CRF"):
        write_video([b"packet"], tmp_path / "invalid.mp4", crf=crf)
    assert not (tmp_path / "invalid.mp4").exists()


def test_write_rejects_empty_packets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)
    with pytest.raises(QRVideoError, match="without QR packets"):
        write_video([], tmp_path / "empty.mp4")


def test_write_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"keep this file")
    with pytest.raises(QRVideoError, match="already exists"):
        write_video([b"packet"], destination)
    assert destination.read_bytes() == b"keep this file"


def test_write_rejects_dangling_destination_symlinks(tmp_path: Path) -> None:
    destination = tmp_path / "link.mp4"
    destination.symlink_to(tmp_path / "missing.mp4")
    with pytest.raises(QRVideoError, match="already exists"):
        write_video([b"packet"], destination)
    assert destination.is_symlink()


def test_missing_ffmpeg_is_diagnosed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(QRVideoError, match="ffmpeg.*not found"):
        write_video([b"packet"], tmp_path / "missing.mp4")


def test_missing_ffprobe_is_diagnosed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(
        video.shutil, "which", lambda name: sys.executable if name == "ffmpeg" else None
    )
    with pytest.raises(QRVideoError, match="ffprobe.*not found"):
        list(read_video(source))


def test_read_rejects_nonlocal_sources() -> None:
    with pytest.raises(QRVideoError, match="existing local file"):
        list(read_video(Path("https://example.invalid/source.mp4")))


def test_probe_uses_local_only_inputs_and_an_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert arguments[arguments.index("-protocol_whitelist") + 1] == "file,pipe"
        assert arguments[arguments.index("-format_whitelist") + 1] == (
            "mov,matroska,webm,avi,mpegts"
        )
        assert arguments[-1] == str(tmp_path / "-https:input.mp4")
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert not kwargs.get("shell")
        return subprocess.CompletedProcess(
            arguments, 0, b'{"streams": [{"width": 640, "height": 360}]}', b""
        )

    monkeypatch.setattr(video.subprocess, "run", run)
    assert video._probe_video(Path("-https:input.mp4")) == (640, 360)


def test_decoder_uses_local_only_inputs_and_a_resolved_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "actual.mp4"
    source.touch()
    (tmp_path / "-https:input.mp4").symlink_to(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)
    process = MagicMock()
    process.read_frame.return_value = None

    def probe(path: Path) -> tuple[int, int]:
        assert path == source.resolve()
        return 640, 360

    def start(arguments: list[str], *, writing: bool) -> Any:
        assert not writing
        input_index = arguments.index("-i")
        assert arguments[input_index + 1] == str(source.resolve())
        assert arguments.index("-protocol_whitelist") < input_index
        assert arguments[arguments.index("-protocol_whitelist") + 1] == "file,pipe"
        assert arguments.index("-format_whitelist") < input_index
        assert arguments[arguments.index("-format_whitelist") + 1] == (
            "mov,matroska,webm,avi,mpegts"
        )
        return process

    monkeypatch.setattr(video, "_probe_video", probe)
    monkeypatch.setattr(video, "_Process", start)
    assert list(read_video(Path("-https:input.mp4"))) == []
    process.finish.assert_called_once_with()
    process.close.assert_called_once_with()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"streams": []},
        {"streams": [{"height": 360}]},
        {"streams": [{"width": "640", "height": 360}]},
        {"streams": [{"width": True, "height": 360}]},
        {"streams": [{"width": 0, "height": 360}]},
        {"streams": [{"width": 640, "height": -1}]},
        {"streams": [{"width": 8193, "height": 1}]},
        {"streams": [{"width": 5000, "height": 2000}]},
    ],
)
def test_probe_rejects_invalid_or_unbounded_geometry(
    metadata: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)
    result = subprocess.CompletedProcess([], 0, json.dumps(metadata).encode(), b"")
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(QRVideoError, match="dimensions"):
        video._probe_video(Path("source.mp4"))


def test_probe_failure_is_diagnosed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video, "_executable", lambda name: sys.executable)
    result = subprocess.CompletedProcess([], 1, b"", b"invalid video container")
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(QRVideoError, match="invalid video container"):
        video._probe_video(Path("source.mp4"))


def test_write_reports_codec_exit_and_drains_large_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    children = _fake_codec(
        monkeypatch,
        "import sys\n"
        "sys.stderr.buffer.write(b'x' * 2_000_000)\n"
        "sys.stderr.flush()\n"
        "sys.stdin.buffer.read()\n"
        "sys.stderr.write('encoder failure')\n"
        "sys.exit(19)\n",
    )
    with pytest.raises(QRVideoError, match="exit status 19.*encoder failure"):
        write_video([b"packet"], tmp_path / "failure.mp4")
    assert len(children) == 1
    assert children[0].poll() == 19


def test_write_reports_a_broken_codec_pipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    children = _fake_codec(
        monkeypatch, "import sys\nsys.stderr.write('input rejected')\nsys.exit(17)\n"
    )
    with pytest.raises(QRVideoError, match="FFmpeg"):
        write_video(_packets(4), tmp_path / "failure.mp4")
    assert len(children) == 1
    assert children[0].poll() is not None


def test_source_generator_error_is_not_replaced_by_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    children = _fake_codec(monkeypatch, "import sys\nsys.stdin.buffer.read()\n")
    failure = OSError("source generator failed")

    def packets() -> Iterator[bytes]:
        yield b"left"
        yield b"right"
        raise failure

    with pytest.raises(OSError) as raised:
        write_video(packets(), tmp_path / "failure.mp4")
    assert raised.value is failure
    assert len(children) == 1
    assert children[0].poll() is not None


def test_read_rejects_a_truncated_raw_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    children = _fake_codec(monkeypatch, "import sys\nsys.stdout.buffer.write(b'x' * 100)\n")
    with pytest.raises(QRVideoError, match="truncated raw frame"):
        list(read_video(source))
    assert children[0].poll() is not None


def test_read_reports_codec_failure_even_after_valid_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    children = _fake_codec(
        monkeypatch,
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff' * (640 * 360))\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('decoder failure')\n"
        "sys.exit(21)\n",
    )
    with pytest.raises(QRVideoError, match="exit status 21.*decoder failure"):
        list(read_video(source))
    assert children[0].poll() == 21


def test_closing_reader_early_stops_only_its_codec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    children = _fake_codec(
        monkeypatch, "import os\nwhile True:\n    os.write(1, b'\\xff' * (640 * 360))\n"
    )
    monkeypatch.setattr(video, "_decode_frame", lambda frame, profile: iter([b"packet"]))
    packets = read_video(source)
    assert isinstance(packets, Generator)
    assert next(packets) == b"packet"
    assert children[0].poll() is None
    packets.close()
    assert children[0].poll() is not None
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


def test_closing_reader_after_transport_error_stops_codec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    children = _fake_codec(
        monkeypatch, "import os\nwhile True:\n    os.write(1, b'\\xff' * (640 * 360))\n"
    )
    monkeypatch.setattr(video, "_decode_frame", lambda frame, profile: iter([b"packet"]))
    with (
        pytest.raises(QRVideoError, match="max-bytes"),
        closing(read_video(source)) as packets,
    ):
        assert next(packets) == b"packet"
        assert children[0].poll() is None
        raise QRVideoError("max-bytes guard stopped decoding")
    assert children[0].poll() is not None
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


@pytest.fixture
def video_tools() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("FFmpeg and ffprobe are required for real video tests")
    return ffmpeg, ffprobe


@pytest.mark.video
@pytest.mark.parametrize(
    ("name", "content"),
    [
        (
            "playlist.m3u8",
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nsegment.ts\n#EXT-X-ENDLIST\n",
        ),
        ("playlist.ffconcat", "ffconcat version 1.0\nfile 'segment.mp4'\n"),
    ],
)
def test_read_rejects_local_playlists_before_opening_segments(
    name: str, content: str, video_tools: tuple[str, str], tmp_path: Path
) -> None:
    source = tmp_path / name
    source.write_text(content)
    with pytest.raises(QRVideoError, match="not on whitelist"):
        list(read_video(source))


@pytest.mark.video
@pytest.mark.parametrize("packet_count", [1, 2, 7])
def test_lossy_h264_round_trip_and_actual_frame_count(
    packet_count: int, video_tools: tuple[str, str], tmp_path: Path
) -> None:
    _, ffprobe = video_tools
    packets = _packets(packet_count)
    destination = tmp_path / "round trip.mp4"
    count = write_video(iter(packets), destination)
    assert count == (packet_count + 1) // 2
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    metadata = json.loads(result.stdout)["streams"][0]
    assert metadata["codec_name"] == "h264"
    assert metadata["pix_fmt"] == "yuv420p"
    assert (metadata["width"], metadata["height"]) == (640, 360)
    assert metadata["avg_frame_rate"] == "30/1"
    assert int(metadata["nb_read_frames"]) == count * CONSERVATIVE.repeat
    expected = [
        packet
        for start in range(0, len(packets), 2)
        for _ in range(CONSERVATIVE.repeat)
        for packet in packets[start : start + 2]
    ]
    assert list(read_video(destination)) == expected


@pytest.mark.video
def test_second_lossy_transcode_preserves_382_byte_binary_packets(
    video_tools: tuple[str, str], tmp_path: Path
) -> None:
    ffmpeg, _ = video_tools
    packets = [b"\x00" * 382, *_packets(6)]
    source, transcoded = tmp_path / "first.mp4", tmp_path / "second.mp4"
    assert write_video(packets, source) == 4
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(transcoded),
        ],
        check=True,
        capture_output=True,
    )
    different_timing = replace(CONSERVATIVE, fps=60, repeat=1)
    decoded = list(read_video(transcoded, profile=different_timing))
    assert list(dict.fromkeys(decoded)) == packets


@pytest.mark.video
def test_scaled_video_and_configurable_timing(video_tools: tuple[str, str], tmp_path: Path) -> None:
    ffmpeg, _ = video_tools
    packets = _packets(4)
    source, scaled = tmp_path / "first.mp4", tmp_path / "scaled.mp4"
    profile = replace(CONSERVATIVE, fps=24, repeat=2)
    assert write_video(packets, source, profile=profile) == 2
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(source),
            "-vf",
            "scale=1280:720:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(scaled),
        ],
        check=True,
        capture_output=True,
    )
    assert list(dict.fromkeys(read_video(scaled))) == packets
