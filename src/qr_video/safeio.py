"""Private atomic writes through canonical parents, with output-leaf symlinks rejected."""

import os
import secrets
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from qr_video.errors import QRVideoError

KEY_SIZE = 32


def _canonical_output(path: Path) -> Path:
    """Resolve directory aliases without following an output-leaf symlink."""
    try:
        if not path.parent.is_dir():
            raise QRVideoError(f"Output directory does not exist: {path.parent}")
        return path.parent.resolve(strict=True) / path.name
    except (OSError, ValueError, RuntimeError) as exc:
        raise QRVideoError(f"Cannot use output directory {path.parent}: {exc}") from exc


def check_output(path: Path, *, overwrite: bool = False, protected: Sequence[Path] = ()) -> None:
    """Allow directory aliases, but reject leaf symlinks and unwanted overwrites."""
    path = _canonical_output(path)
    try:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and stat.S_ISLNK(mode):
            raise QRVideoError(f"Refusing to write through a symlink: {path}")
        resolved = path.resolve()
        for other in protected:
            if resolved == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                raise QRVideoError(f"Output aliases a protected file: {other}")
        if mode is None:
            return
        if not stat.S_ISREG(mode):
            raise QRVideoError(f"Output is not a regular file: {path}")
        if not overwrite:
            raise QRVideoError(f"Output already exists: {path}")
    except (OSError, ValueError, RuntimeError) as exc:
        raise QRVideoError(f"Cannot use output path {path}: {exc}") from exc


@contextmanager
def atomic_output(path: Path, *, overwrite: bool = False) -> Iterator[BinaryIO]:
    """Write a mode-0600 sibling, then link without clobbering or explicitly replace."""
    path = _canonical_output(path)
    check_output(path, overwrite=overwrite)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".qr-video-", suffix=".part", dir=path.parent)
        temporary = Path(name)
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o600)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        check_output(path, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except OSError as exc:
        raise QRVideoError(f"Cannot write {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise QRVideoError(f"Cannot remove private output {temporary}: {exc}") from exc


@contextmanager
def open_input(path: Path) -> Iterator[BinaryIO]:
    """Open a regular file without blocking on a FIFO."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise QRVideoError(f"Input is not a regular file: {path}")
            yield stream
    except (OSError, ValueError) as exc:
        raise QRVideoError(f"Cannot read {path}: {exc}") from exc


def read_key(path: Path) -> bytes:
    """Read exactly 32 raw bytes, not a password, hex string, or encoded key."""
    with open_input(path) as stream:
        key = stream.read(KEY_SIZE + 1)
    if len(key) != KEY_SIZE:
        raise QRVideoError(f"Key must contain exactly {KEY_SIZE} raw bytes: {path}")
    return key


def generate_key(path: Path) -> None:
    """Generate and atomically publish a private key without replacing any file."""
    with atomic_output(path) as stream:
        stream.write(secrets.token_bytes(KEY_SIZE))
