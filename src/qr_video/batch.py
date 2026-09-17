import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from qr_video.errors import QRVideoError


@dataclass(frozen=True)
class BatchItem:
    source: Path
    destination: Path


@dataclass(frozen=True)
class BatchNotice:
    source: Path
    reason: str


@dataclass(frozen=True)
class BatchPlan:
    items: tuple[BatchItem, ...]
    skipped: tuple[BatchNotice, ...]
    failures: tuple[BatchNotice, ...]
    protected: tuple[Path, ...]


def _destination(source: Path, output_dir: Path, *, decode: bool) -> Path:
    name = source.name[:-4] if decode else source.name + ".mp4"
    if name in ("", ".", ".."):
        raise QRVideoError(f"Cannot derive an output filename from {source.name!r}")
    return output_dir / name


def prepare_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    decode: bool = False,
    key_file: Path | None = None,
) -> BatchPlan:
    try:
        if not input_dir.is_dir():
            raise QRVideoError(f"Input is not a directory: {input_dir}")
        input_dir = input_dir.resolve(strict=True)
        output_dir = output_dir.resolve()
        if input_dir == output_dir or (output_dir.exists() and input_dir.samefile(output_dir)):
            raise QRVideoError("Input and output directories must be different")
        if output_dir.exists() and not output_dir.is_dir():
            raise QRVideoError(f"Output is not a directory: {output_dir}")
        sources = sorted(input_dir.iterdir(), key=lambda path: path.name)
    except (OSError, ValueError, RuntimeError) as exc:
        raise QRVideoError(f"Cannot prepare batch directories: {exc}") from exc

    items: list[BatchItem] = []
    skipped: list[BatchNotice] = []
    failures: list[BatchNotice] = []
    protected = [key_file] if key_file is not None else []
    for source in sources:
        try:
            if not stat.S_ISREG(source.stat().st_mode):
                skipped.append(BatchNotice(source, "not a regular file"))
                continue
            protected.append(source)
            if key_file is not None and source.samefile(key_file):
                skipped.append(BatchNotice(source, "selected key file"))
                continue
            if decode and not source.name.lower().endswith(".mp4"):
                skipped.append(BatchNotice(source, "filename does not end in .mp4"))
                continue
            items.append(BatchItem(source, _destination(source, output_dir, decode=decode)))
        except (QRVideoError, OSError) as exc:
            failures.append(BatchNotice(source, str(exc)))

    counts = Counter(item.destination for item in items)
    unique_items: list[BatchItem] = []
    for item in items:
        if counts[item.destination] > 1:
            failures.append(
                BatchNotice(item.source, f"Conflicting output filename: {item.destination}")
            )
        else:
            unique_items.append(item)

    if unique_items:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir.resolve(strict=True)
            if input_dir.samefile(output_dir):
                raise QRVideoError("Input and output directories must be different")
            with tempfile.TemporaryFile(dir=output_dir):
                pass
        except (OSError, ValueError, RuntimeError) as exc:
            raise QRVideoError(f"Cannot use output directory {output_dir}: {exc}") from exc
        unique_items = [
            BatchItem(item.source, output_dir / item.destination.name) for item in unique_items
        ]
    return BatchPlan(tuple(unique_items), tuple(skipped), tuple(failures), tuple(protected))
