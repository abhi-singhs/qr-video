import argparse
import json
import shutil
import sys
import tempfile
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

from qr_video import __version__
from qr_video.envelope import EnvelopeInfo, decode_envelope, encode_envelope
from qr_video.errors import QRVideoError
from qr_video.profiles import CONSERVATIVE, Profile
from qr_video.safeio import (
    atomic_output,
    check_output,
    generate_key,
    read_key,
)
from qr_video.transport import (
    DEFAULT_MAX_BYTES,
    Manifest,
    encode_packets,
    prepare_manifest,
    recover_stream,
    transport_stats,
    validate_coding,
)
from qr_video.video import read_video, write_video


class _Progress:
    def __init__(
        self,
        label: str,
        unit: str,
        *,
        total: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.label = label
        self.unit = unit
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.enabled = self.stream.isatty()
        self._width = 0
        if self.enabled:
            self.update(0)

    def update(self, completed: int) -> None:
        if not self.enabled:
            return
        if self.total is None:
            marker = "|/-\\"[completed % 4]
            message = f"{self.label}: {marker} {completed} {self.unit}"
        else:
            bounded = min(completed, self.total)
            percent = 100 * bounded // self.total
            message = f"{self.label}: {percent:3d}% ({bounded}/{self.total} {self.unit})"
        self._width = max(self._width, len(message))
        self.stream.write(f"\r{message:<{self._width}}")
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _inline_key(value: str) -> bytes:
    key = value.encode("utf-8")
    if not key:
        raise argparse.ArgumentTypeError("key must not be empty")
    return key


def _add_key_arguments(command: argparse.ArgumentParser) -> None:
    keys = command.add_mutually_exclusive_group()
    keys.add_argument(
        "--key",
        type=_inline_key,
        help="Use a non-empty key directly (encoded as UTF-8)",
    )
    keys.add_argument("--key-file", type=Path, help="Read a non-empty raw key from a file")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="qr-video",
        description="Encode local binary files into QR-code MP4 videos and recover them.",
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser(
        "keygen", help="Create a private 32-character alphanumeric key file"
    )
    keygen.add_argument("--out", required=True, type=Path, help="New key path, never overwritten")
    for command in ("encode", "stats"):
        help_text = (
            "Write a local QR-code MP4"
            if command == "encode"
            else ("Calculate exact framing, duration and payload rate without writing video")
        )
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("input", type=Path, help="Local binary input file")
        _add_key_arguments(sub)
        sub.add_argument(
            "--compress", action="store_true", help="Compress with zlib before encryption"
        )
        sub.add_argument("--profile", choices=["conservative"], default="conservative")
        sub.add_argument("--fps", type=int, default=30, help="Video FPS, default 30, maximum 120")
        sub.add_argument(
            "--repeat", type=int, default=3, help="Frames per QR image, must divide FPS"
        )
        sub.add_argument("--data-shards", type=int, default=100, help="Data shards per full block")
        sub.add_argument(
            "--parity-shards", type=int, default=25, help="Parity shards per full block"
        )
        sub.add_argument("--interleave", type=int, default=4, help="Interleaved blocks, 1 to 16")
        if command == "encode":
            sub.add_argument("--out", required=True, type=Path, help="Output MP4 path")
            sub.add_argument(
                "--overwrite", action="store_true", help="Replace an existing output file"
            )
            sub.add_argument("--crf", type=int, default=18, help="H.264 CRF, 0 to 51, default 18")
    decode = commands.add_parser("decode", help="Recover a binary file from a local video")
    decode.add_argument("input", type=Path, help="Local input video")
    decode.add_argument("--out", required=True, type=Path, help="Recovered binary output path")
    _add_key_arguments(decode)
    decode.add_argument(
        "--overwrite", action="store_true", help="Replace output only after verification"
    )
    decode.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Limit both envelope and original bytes, default 1073741824, maximum 68719476736",
    )
    return root


def _input_and_key(args: argparse.Namespace) -> tuple[Path, bytes | None]:
    source: Path = args.input
    if not source.is_file():
        raise QRVideoError(f"Input is not a regular file: {source}")
    key_file: Path | None = args.key_file
    if key_file is not None and key_file.exists() and source.samefile(key_file):
        raise QRVideoError("Input and key file must be different files")
    key: bytes | None = args.key
    return source, read_key(key_file) if key_file is not None else key


def _profile(args: argparse.Namespace) -> Profile:
    profile = replace(CONSERVATIVE, fps=args.fps, repeat=args.repeat)
    profile.validate()
    validate_coding(args.data_shards, args.parity_shards, args.interleave)
    if args.command == "encode" and not 0 <= args.crf <= 51:
        raise QRVideoError("CRF must be between 0 and 51")
    return profile


def _check_destination(args: argparse.Namespace, source: Path) -> None:
    args.out = args.out.parent.resolve(strict=True) / args.out.name
    protected = [source]
    if args.key_file is not None:
        protected.append(args.key_file)
    check_output(args.out, overwrite=args.overwrite, protected=protected)


def _statistics(
    manifest: Manifest,
    info: EnvelopeInfo,
    profile: Profile,
) -> dict[str, str | int | float | bool]:
    stats: dict[str, str | int | float | bool] = {
        "profile": profile.name,
        "width": profile.width,
        "height": profile.height,
        "fps": profile.fps,
        "repeat": profile.repeat,
        "images_per_second": profile.fps / profile.repeat,
        "encrypted": info.encrypted,
        "compressed": info.compressed,
        "original_bytes": info.original_size,
        "data_shards_per_full_block": manifest.k,
        "parity_shards_per_full_block": manifest.m,
        "interleave_blocks": manifest.interleave,
    }
    stats.update(transport_stats(manifest, profile))
    stats["application_mb_per_hour"] = (
        info.original_size / float(stats["duration_seconds"]) * 3600 / 1_000_000
    )
    return stats


def _encode_or_stats(args: argparse.Namespace) -> None:
    profile = _profile(args)
    source, key = _input_and_key(args)
    if args.command == "encode":
        _check_destination(args, source)
    with tempfile.TemporaryDirectory(prefix="qr-video-encode-") as directory:
        workspace = Path(directory).resolve()
        envelope = workspace / "stream.qve"
        info = encode_envelope(source, envelope, key=key, compress=args.compress)
        manifest = prepare_manifest(
            envelope,
            k=args.data_shards,
            m=args.parity_shards,
            interleave=args.interleave,
        )
        stats = _statistics(manifest, info, profile)
        if args.command == "stats":
            print(json.dumps(stats, indent=2))
            return
        print(json.dumps(stats, sort_keys=True), file=sys.stderr, flush=True)
        video = workspace / "video.mp4"
        progress = _Progress("Encoding", "QR images", total=int(stats["images"]))
        try:
            images = write_video(
                encode_packets(envelope, manifest),
                video,
                profile=profile,
                crf=args.crf,
                on_progress=progress.update,
            )
        finally:
            progress.close()
        if images != stats["images"]:
            raise QRVideoError("Video image count does not match the transport schedule")
        with (
            video.open("rb") as source_video,
            atomic_output(
                args.out,
                overwrite=args.overwrite,
            ) as output,
        ):
            shutil.copyfileobj(source_video, output, length=1024**2)
    print(f"Encoded {info.original_size} bytes to {args.out}")


def _decode(args: argparse.Namespace) -> None:
    source, key = _input_and_key(args)
    _check_destination(args, source)
    with tempfile.TemporaryDirectory(prefix="qr-video-decode-") as directory:
        envelope = Path(directory).resolve() / "stream.qve"
        progress = _Progress("Decoding", "video frames")
        try:
            with closing(read_video(source, on_progress=progress.update)) as observations:
                _, recovery = recover_stream(observations, envelope, max_bytes=args.max_bytes)
        finally:
            progress.close()
        info = decode_envelope(
            envelope,
            args.out,
            key=key,
            overwrite=args.overwrite,
            max_bytes=args.max_bytes,
        )
    print(json.dumps(asdict(recovery), sort_keys=True), file=sys.stderr)
    print(f"Recovered {info.original_size} bytes to {args.out}; SHA-256 {info.sha256}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "keygen":
            generate_key(args.out)
            print(f"Created key file {args.out}. Keep it private; losing it prevents decryption.")
        elif args.command in ("encode", "stats"):
            _encode_or_stats(args)
        else:
            _decode(args)
    except (QRVideoError, OSError) as exc:
        print(f"qr-video: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("qr-video: interrupted; no incomplete output published", file=sys.stderr)
        return 130
    return 0
