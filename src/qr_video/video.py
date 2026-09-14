import json
import os
import shutil
import subprocess
import threading
from collections.abc import Generator, Iterable, Iterator
from contextlib import suppress
from itertools import zip_longest
from pathlib import Path
from typing import IO, Any

import numpy as np
import qrcode
import qrcode.base
import zxingcpp
from numpy.typing import NDArray
from qrcode.exceptions import DataOverflowError
from qrcode.util import MODE_8BIT_BYTE, BitBuffer, QRData, create_bytes

from qr_video.errors import QRVideoError
from qr_video.profiles import CONSERVATIVE, Profile

_MAX_DIMENSION = 8192
_MAX_FRAME_PIXELS = 3840 * 2160
_STDERR_LIMIT = 65536
_PROCESS_TIMEOUT = 60
_INPUT_FORMATS = "mov,matroska,webm,avi,mpegts"
GrayFrame = NDArray[np.uint8]


def _zero_safe_codewords(packet: bytes, profile: Profile) -> list[int]:
    # qrcode 8 raises glog(0) for an all-zero Reed-Solomon data block.
    blocks = qrcode.base.rs_blocks(profile.version, qrcode.constants.ERROR_CORRECT_H)
    capacity = sum(block.data_count for block in blocks)
    buffer = BitBuffer()
    buffer.put(MODE_8BIT_BYTE, 4)
    buffer.put(len(packet), 16)
    QRData(packet, mode=MODE_8BIT_BYTE).write(buffer)
    buffer.put(0, 4)
    for index in range(capacity - len(buffer.buffer)):
        buffer.put(0xEC if index % 2 == 0 else 0x11, 8)
    data: list[list[int]] = []
    parity: list[list[int]] = []
    offset = 0
    for block in blocks:
        count = block.data_count
        chunk = BitBuffer()
        chunk.buffer = buffer.buffer[offset : offset + count]
        offset += count
        if any(chunk.buffer):
            encoded = create_bytes(chunk, [block])
            parity.append(encoded[count:])
        else:
            parity.append([0] * (block.total_count - count))
        data.append(chunk.buffer)
    return [
        value
        for group in (data, parity)
        for column in zip_longest(*group)
        for value in column
        if value is not None
    ]


def _validate_profile(profile: Profile) -> None:
    if any(
        type(value) is not int
        for value in (
            profile.identifier,
            profile.version,
            profile.pixels_per_module,
            profile.qr_capacity,
            profile.width,
            profile.height,
            profile.fps,
            profile.repeat,
        )
    ):
        raise QRVideoError("Profile geometry, FPS, and frame repeat must be integers")
    profile.validate()


def _render_qr(packet: bytes, profile: Profile) -> GrayFrame:
    if not isinstance(packet, bytes):
        raise QRVideoError("QR packets must be bytes")
    if len(packet) > profile.qr_capacity:
        raise QRVideoError(
            f"QR packet contains {len(packet)} bytes; maximum is {profile.qr_capacity}"
        )
    qr = qrcode.QRCode(
        version=profile.version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=profile.pixels_per_module,
        border=4,
    )
    qr.add_data(QRData(packet, mode=MODE_8BIT_BYTE), optimize=0)
    try:
        qr.make(fit=False)
    except DataOverflowError as error:
        raise QRVideoError("QR packet exceeds the fixed byte-mode capacity") from error
    except ValueError as error:
        if str(error) != "glog(0)":
            raise QRVideoError(f"Could not encode the QR packet. {error}") from error
        qr.data_cache = _zero_safe_codewords(packet, profile)
        qr.make(fit=False)
    image = np.asarray(
        qr.make_image(fill_color="black", back_color="white").convert("L"), dtype=np.uint8
    )
    if image.shape != (profile.side, profile.side):
        raise QRVideoError("QR image dimensions do not match the conservative profile")
    return image


def render_frame(left: bytes, right: bytes | None, profile: Profile = CONSERVATIVE) -> GrayFrame:
    _validate_profile(profile)
    frame = np.full((profile.height, profile.width), 255, dtype=np.uint8)
    for packet, (x, y) in zip((left, right), profile.positions, strict=True):
        if packet is not None:
            frame[y : y + profile.side, x : x + profile.side] = _render_qr(packet, profile)
    if left is None:
        raise QRVideoError("The left QR packet must be bytes")
    return frame


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise QRVideoError(f"{name} is required but was not found on PATH")
    return executable


class _Process:
    def __init__(self, arguments: list[str], *, writing: bool) -> None:
        try:
            self.process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL if writing else subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as error:
            raise QRVideoError(f"Could not start FFmpeg. {error}") from error
        self._stderr = bytearray()
        assert self.process.stderr is not None
        self._reader = threading.Thread(
            target=self._drain_stderr, args=(self.process.stderr,), daemon=True
        )
        self._reader.start()

    def _drain_stderr(self, stream: IO[bytes]) -> None:
        try:
            while chunk := stream.read(8192):
                self._stderr.extend(chunk)
                if len(self._stderr) > _STDERR_LIMIT:
                    del self._stderr[:-_STDERR_LIMIT]
        except OSError:
            pass

    def _failure(self, message: str) -> QRVideoError:
        detail = self._stderr.decode("utf-8", errors="replace").strip()
        return QRVideoError(f"{message}. {detail}" if detail else message)

    def write(self, frame: GrayFrame) -> None:
        assert self.process.stdin is not None
        remaining = memoryview(frame).cast("B")
        try:
            while remaining:
                written = os.write(self.process.stdin.fileno(), remaining)
                if written <= 0:
                    raise BrokenPipeError("FFmpeg closed its input pipe")
                remaining = remaining[written:]
        except OSError as error:
            self.close()
            raise self._failure("Could not write a video frame to FFmpeg") from error

    def read_frame(self, size: int) -> bytearray | None:
        assert self.process.stdout is not None
        raw = bytearray()
        try:
            while len(raw) < size:
                chunk = self.process.stdout.read(size - len(raw))
                if not chunk:
                    if not raw:
                        return None
                    raise QRVideoError(
                        f"FFmpeg returned a truncated raw frame ({len(raw)} of {size} bytes)"
                    )
                raw.extend(chunk)
        except OSError as error:
            self.close()
            raise self._failure("Could not read a video frame from FFmpeg") from error
        return raw

    def finish(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError as error:
                self.close()
                raise self._failure("Could not finish writing to FFmpeg") from error
        try:
            returncode = self.process.wait(timeout=_PROCESS_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            self.close()
            raise self._failure("FFmpeg did not finish within 60 seconds") from error
        self._reader.join()
        if returncode:
            raise self._failure(f"FFmpeg failed with exit status {returncode}")

    def close(self) -> None:
        if self.process.poll() is None:
            with suppress(OSError):
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    self.process.kill()
                with suppress(subprocess.TimeoutExpired, OSError):
                    self.process.wait(timeout=5)
            except OSError:
                pass
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
        self._reader.join(timeout=5)
        if not self._reader.is_alive() and self.process.stderr is not None:
            with suppress(OSError):
                self.process.stderr.close()


def write_video(
    packets: Iterable[bytes],
    destination: Path,
    *,
    profile: Profile = CONSERVATIVE,
    crf: int = 18,
) -> int:
    _validate_profile(profile)
    if type(crf) is not int or not 0 <= crf <= 51:
        raise QRVideoError("CRF must be an integer between 0 and 51")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise QRVideoError("Video destination already exists")
    if not destination.parent.is_dir():
        raise QRVideoError("Video destination directory does not exist")
    executable = _executable("ffmpeg")
    iterator = iter(packets)
    try:
        left = next(iterator)
    except StopIteration as error:
        raise QRVideoError("Cannot write a video without QR packets") from error
    frame = render_frame(left, next(iterator, None), profile)
    process = _Process(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "gray",
            "-video_size",
            f"{profile.width}x{profile.height}",
            "-framerate",
            str(profile.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            str(destination),
        ],
        writing=True,
    )
    count = 0
    try:
        while True:
            for _ in range(profile.repeat):
                process.write(frame)
            count += 1
            try:
                left = next(iterator)
            except StopIteration:
                break
            frame = render_frame(left, next(iterator, None), profile)
        process.finish()
    finally:
        process.close()
    return count


def _probe_video(source: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                _executable("ffprobe"),
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                _INPUT_FORMATS,
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source.absolute()),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise QRVideoError("ffprobe did not finish within 30 seconds") from error
    except OSError as error:
        raise QRVideoError(f"Could not start ffprobe. {error}") from error
    if result.returncode:
        detail = result.stderr[-_STDERR_LIMIT:].decode("utf-8", errors="replace").strip()
        raise QRVideoError(f"ffprobe could not read the video. {detail}")
    try:
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        width, height = stream["width"], stream["height"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise QRVideoError("ffprobe did not report a video stream with valid dimensions") from error
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= _MAX_DIMENSION
        or not 1 <= height <= _MAX_DIMENSION
        or width * height > _MAX_FRAME_PIXELS
    ):
        raise QRVideoError("Video dimensions exceed the supported limit of 8,294,400 pixels")
    return width, height


def _read_qrs(image: GrayFrame) -> Iterator[bytes]:
    results = zxingcpp.read_barcodes(
        image,
        formats=zxingcpp.BarcodeFormats(zxingcpp.BarcodeFormat.QRCode),
        try_downscale=False,
        try_invert=False,
    )
    for result in results:
        packet: Any = result.bytes
        if not isinstance(packet, bytes):
            raise QRVideoError("The QR decoder must provide raw packet bytes")
        yield packet


def _decode_frame(frame: GrayFrame, profile: Profile) -> Iterator[bytes]:
    height, width = frame.shape
    seen: set[bytes] = set()
    found = 0
    for x, y in profile.positions:
        x0 = round(x * width / profile.width)
        y0 = round(y * height / profile.height)
        x1 = round((x + profile.side) * width / profile.width)
        y1 = round((y + profile.side) * height / profile.height)
        crop = frame[y0:y1, x0:x1]
        if not crop.size:
            continue
        packets = list(_read_qrs(crop))
        if packets:
            found += 1
        for packet in packets:
            seen.add(packet)
            yield packet
    if found < 2:
        for packet in _read_qrs(frame):
            if packet not in seen:
                seen.add(packet)
                yield packet


def read_video(source: Path, *, profile: Profile = CONSERVATIVE) -> Generator[bytes, None, None]:
    _validate_profile(profile)
    try:
        source = source.resolve()
    except (OSError, RuntimeError) as error:
        raise QRVideoError("Video source must be an existing local file") from error
    if not source.is_file():
        raise QRVideoError("Video source must be an existing local file")
    executable = _executable("ffmpeg")
    width, height = _probe_video(source)
    process = _Process(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            _INPUT_FORMATS,
            "-noautorotate",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale={width}:{height}:flags=neighbor",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        writing=False,
    )
    try:
        while (raw := process.read_frame(width * height)) is not None:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
            yield from _decode_frame(frame, profile)
        process.finish()
    finally:
        process.close()
