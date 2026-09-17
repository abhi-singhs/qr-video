import argparse
import os
from collections.abc import Callable, Generator, Iterable, Sequence
from pathlib import Path

import pytest

from qr_video import cli as cli_module
from qr_video.batch import prepare_batch
from qr_video.envelope import encode_envelope
from qr_video.errors import QRVideoError
from qr_video.profiles import Profile
from qr_video.transport import encode_packets, prepare_manifest

NAMES = (
    ".hidden",
    "README",
    "archive.tar.gz",
    "caf\u00e9.txt",
    "clip.mp4",
    "report.pdf",
    "report.txt",
    "two words.txt",
)


@pytest.mark.parametrize("decode", [False, True])
def test_batch_preserves_complete_names(tmp_path: Path, decode: bool) -> None:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for name in NAMES:
        (source_dir / (name + ".mp4" if decode else name)).write_bytes(b"fixture")
    output_dir = tmp_path / "new" / "output"
    plan = prepare_batch(source_dir, output_dir, decode=decode)
    assert [item.source.name for item in plan.items] == sorted(
        name + ".mp4" if decode else name for name in NAMES
    )
    assert {item.destination for item in plan.items} == {
        output_dir / (name if decode else name + ".mp4") for name in NAMES
    }
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not plan.failures
    assert not plan.skipped


def test_batch_snapshot_is_top_level_only(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b").write_bytes(b"b")
    (tmp_path / "linked-dir").symlink_to(tmp_path / "nested", target_is_directory=True)
    (tmp_path / "linked-file").symlink_to(tmp_path / "a")
    plan = prepare_batch(tmp_path, tmp_path / "output")
    (tmp_path / "later").write_bytes(b"later")
    assert [item.source.name for item in plan.items] == ["a", "linked-file"]
    assert {notice.source.name for notice in plan.skipped} == {"nested", "linked-dir"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO")
def test_batch_skips_fifo_without_opening_it(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "fifo")
    plan = prepare_batch(tmp_path, tmp_path / "output")
    assert not plan.items
    assert plan.skipped[0].source.name == "fifo"
    assert not (tmp_path / "output").exists()


def test_decode_selection_and_invalid_names(tmp_path: Path) -> None:
    for name in ("a.MP4", "notes.txt", ".mp4", "..mp4", "...mp4"):
        (tmp_path / name).write_bytes(b"fixture")
    plan = prepare_batch(tmp_path, tmp_path / "output", decode=True)
    assert [item.destination.name for item in plan.items] == ["a"]
    assert [notice.source.name for notice in plan.skipped] == ["notes.txt"]
    assert {notice.source.name for notice in plan.failures} == {".mp4", "..mp4", "...mp4"}
    assert all("Cannot derive" in notice.reason for notice in plan.failures)
    assert tmp_path / "notes.txt" in plan.protected


def test_conflicting_derived_names_fail_both_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lower = tmp_path / "same.mp4"
    upper = tmp_path / "same.MP4"
    lower.write_bytes(b"fixture")
    upper.write_bytes(b"fixture")
    # Model a case-sensitive input folder even on case-insensitive hosts.
    monkeypatch.setattr(Path, "iterdir", lambda self: iter((lower, upper)))
    plan = prepare_batch(tmp_path, tmp_path / "output", decode=True)
    assert not plan.items
    assert {notice.source for notice in plan.failures} == {lower, upper}
    assert all("Conflicting output" in notice.reason for notice in plan.failures)


def test_unreadable_entry_is_an_individual_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad"
    bad.write_bytes(b"fixture")
    (tmp_path / "good").write_bytes(b"fixture")
    original_stat = Path.stat

    def stat_path(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == bad:
            raise PermissionError("injected stat failure")
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat_path)
    plan = prepare_batch(tmp_path, tmp_path / "output")
    assert [item.source.name for item in plan.items] == ["good"]
    assert len(plan.failures) == 1
    assert plan.failures[0].source == bad
    assert "injected stat failure" in plan.failures[0].reason


@pytest.mark.parametrize("kind", ["same", "alias", "missing-input", "file-output"])
def test_invalid_batch_roots(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    if kind == "same":
        output = source
    elif kind == "alias":
        output.symlink_to(source, target_is_directory=True)
    elif kind == "missing-input":
        source = tmp_path / "missing"
    else:
        output.write_bytes(b"keep")
    with pytest.raises(QRVideoError, match="director"):
        prepare_batch(source, output)
    if kind == "file-output":
        assert output.read_bytes() == b"keep"


@pytest.fixture
def fake_video(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    workspaces: list[Path] = []

    def write_video(
        packets: Iterable[bytes],
        destination: Path,
        *,
        profile: Profile,
        crf: int,
        on_progress: Callable[[int], None],
    ) -> int:
        assert all(not workspace.exists() for workspace in workspaces)
        workspaces.append(destination.parent)
        count = sum(1 for _ in packets)
        destination.write_bytes(b"complete video fixture")
        return (count + 1) // 2

    monkeypatch.setattr(cli_module, "write_video", write_video)
    return workspaces


@pytest.mark.parametrize("overwrite", [False, True])
def test_batch_continues_after_existing_output(
    tmp_path: Path, fake_video: list[Path], capsys: pytest.CaptureFixture[str], overwrite: bool
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("a", "b", "c"):
        (source / name).write_bytes(name.encode())
    (output / "b.mp4").write_bytes(b"keep")
    result = cli_module.main(
        ["encode", "--input-dir", str(source), "--out-dir", str(output)]
        + (["--overwrite"] if overwrite else [])
    )
    assert result == (0 if overwrite else 1)
    assert (output / "a.mp4").read_bytes() == b"complete video fixture"
    assert (output / "c.mp4").read_bytes() == b"complete video fixture"
    assert (output / "b.mp4").read_bytes() == (b"complete video fixture" if overwrite else b"keep")
    assert len(fake_video) == (3 if overwrite else 2)
    assert all(not workspace.exists() for workspace in fake_video)
    captured = capsys.readouterr()
    if overwrite:
        assert "3 succeeded, 0 failed, 0 skipped" in captured.out
    else:
        assert "2 succeeded, 1 failed, 0 skipped" in captured.out
        assert str(source / "b") in captured.err.split("Failed files:")[1]
    assert not list(output.glob(".qr-video-*.part"))


def test_batch_reads_key_once_and_skips_all_key_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_video: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    key = source / "key"
    key.write_bytes(b"test key")
    (source / "key-link").symlink_to(key)
    os.link(key, source / "key-hardlink")
    for name in ("a", "b"):
        (source / name).write_bytes(b"payload")
    output = tmp_path / "output"
    read_key = cli_module.read_key
    key_reads: list[Path] = []

    def read_once(path: Path) -> bytes:
        key_reads.append(path)
        return read_key(path)

    monkeypatch.setattr(cli_module, "read_key", read_once)
    assert (
        cli_module.main(
            ["encode", "--input-dir", str(source), "--out-dir", str(output), "--key-file", str(key)]
        )
        == 0
    )
    assert key_reads == [key]
    assert {path.name for path in output.iterdir()} == {"a.mp4", "b.mp4"}
    captured = capsys.readouterr()
    assert "2 succeeded, 0 failed, 3 skipped" in captured.out
    assert captured.err.count("selected key file") == 3
    assert key.read_bytes() == b"test key"


def test_batch_protects_other_inputs_and_key(
    tmp_path: Path, fake_video: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("a", "b"):
        (source / name).write_bytes(name.encode())
    key = tmp_path / "key"
    key.write_bytes(b"test key")
    os.link(source / "b", output / "a.mp4")
    os.link(key, output / "b.mp4")
    assert (
        cli_module.main(
            [
                "encode",
                "--input-dir",
                str(source),
                "--out-dir",
                str(output),
                "--key-file",
                str(key),
                "--overwrite",
            ]
        )
        == 1
    )
    assert not fake_video
    assert (output / "a.mp4").samefile(source / "b")
    assert (output / "b.mp4").samefile(key)
    captured = capsys.readouterr()
    assert "0 succeeded, 2 failed, 0 skipped" in captured.out
    failures = captured.err.split("Failed files:")[1]
    assert str(source / "a") in failures
    assert str(source / "b") in failures


def test_batch_does_not_overwrite_an_earlier_output_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_video: list[Path]
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for name in ("a", "b"):
        (source / name).write_bytes(name.encode())
    encode = cli_module._encode_or_stats

    def link_after_encoding(
        args: argparse.Namespace,
        *,
        key: bytes | None,
        profile: Profile,
        protected: Sequence[Path] = (),
    ) -> None:
        encode(args, key=key, profile=profile, protected=protected)
        os.link(output / "a.mp4", output / "b.mp4")

    monkeypatch.setattr(cli_module, "_encode_or_stats", link_after_encoding)
    assert (
        cli_module.main(
            ["encode", "--input-dir", str(source), "--out-dir", str(output), "--overwrite"]
        )
        == 1
    )
    assert len(fake_video) == 1
    assert (output / "a.mp4").samefile(output / "b.mp4")


def test_case_distinct_destinations_follow_filesystem_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_video: list[Path]
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    lower, upper = source / "a", source / "A"
    lower.write_bytes(b"lower")
    upper.write_bytes(b"upper")
    aliases = lower.samefile(upper)
    iterdir = Path.iterdir
    monkeypatch.setattr(
        Path, "iterdir", lambda self: iter((lower, upper)) if self == source else iterdir(self)
    )
    assert cli_module.main(
        ["encode", "--input-dir", str(source), "--out-dir", str(output), "--overwrite"]
    ) == (1 if aliases else 0)
    assert len(fake_video) == (1 if aliases else 2)
    assert (output / "a.mp4").samefile(output / "A.mp4") == aliases


def test_batch_output_parent_stays_fixed_for_later_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_video: list[Path]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a", "b"):
        (source / name).write_bytes(name.encode())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    writer = cli_module.write_video

    def switch_alias(
        packets: Iterable[bytes],
        destination: Path,
        *,
        profile: Profile,
        crf: int,
        on_progress: Callable[[int], None] | None = None,
    ) -> int:
        count = writer(packets, destination, profile=profile, crf=crf, on_progress=on_progress)
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        return count

    monkeypatch.setattr(cli_module, "write_video", switch_alias)
    assert cli_module.main(["encode", "--input-dir", str(source), "--out-dir", str(alias)]) == 0
    assert {path.name for path in first.iterdir()} == {"a.mp4", "b.mp4"}
    assert list(second.iterdir()) == []


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("encode", ["--repeat", "0"]),
        ("encode", ["--crf", "52"]),
        ("encode", ["--data-shards", "0"]),
        ("decode", ["--max-bytes", "0"]),
        ("decode", ["--max-bytes", str(64 * 1024**3 + 1)]),
        ("encode", ["--key-file", "missing-key"]),
    ],
)
def test_shared_errors_fail_before_creating_outputs(
    tmp_path: Path, fake_video: list[Path], command: str, options: list[str]
) -> None:
    (tmp_path / "file.mp4").write_bytes(b"fixture")
    output = tmp_path / "output"
    assert (
        cli_module.main([command, "--input-dir", str(tmp_path), "--out-dir", str(output), *options])
        == 1
    )
    assert not output.exists()
    assert not fake_video


def test_empty_batch_key_fails_before_creating_outputs(
    tmp_path: Path, fake_video: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    key = tmp_path / "key"
    key.touch()
    output = tmp_path / "output"
    assert (
        cli_module.main(
            [
                "encode",
                "--input-dir",
                str(tmp_path),
                "--out-dir",
                str(output),
                "--key-file",
                str(key),
            ]
        )
        == 1
    )
    assert "must not be empty" in capsys.readouterr().err
    assert not output.exists()
    assert not fake_video


def test_unwritable_output_directory_fails_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_video: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "source").write_bytes(b"payload")
    output = tmp_path / "output"

    def fail_probe(*, dir: Path) -> None:
        raise PermissionError("injected permission error")

    monkeypatch.setattr("qr_video.batch.tempfile.TemporaryFile", fail_probe)
    assert cli_module.main(["encode", "--input-dir", str(tmp_path), "--out-dir", str(output)]) == 1
    assert "Cannot use output directory" in capsys.readouterr().err
    assert not fake_video
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("command", ["encode", "decode"])
@pytest.mark.parametrize("contents", ["empty", "ineligible", "key-only"])
def test_empty_batch_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, contents: str
) -> None:
    options: list[str] = []
    if contents == "ineligible":
        (tmp_path / "directory").mkdir()
        if command == "decode":
            (tmp_path / "notes").write_bytes(b"not a video")
    elif contents == "key-only":
        key = tmp_path / "key"
        key.write_bytes(b"test key")
        options = ["--key-file", str(key)]
    assert (
        cli_module.main(
            [command, "--input-dir", str(tmp_path), "--out-dir", str(tmp_path / "output"), *options]
        )
        == 1
    )
    assert "No eligible input files" in capsys.readouterr().err


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("failure", ["none", "wrong-key", "corrupt", "limit"])
def test_batch_decode_integrity_and_per_file_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    overwrite: bool,
) -> None:
    videos, output = tmp_path / "videos", tmp_path / "output"
    videos.mkdir()
    output.mkdir()
    envelopes: dict[Path, Path] = {}
    for name in ("a", "b", "c"):
        source = tmp_path / name
        source.write_bytes(b"payload" * (100 if name == "b" and failure == "limit" else 40))
        envelope = tmp_path / (name + ".qve")
        key = b"wrong" if name == "b" and failure == "wrong-key" else b"correct"
        encode_envelope(source, envelope, key=key)
        if name == "b" and failure == "corrupt":
            data = bytearray(envelope.read_bytes())
            data[-1] ^= 1
            envelope.write_bytes(data)
        video = videos / (name + ".bin.mp4")
        video.write_bytes(b"fake video")
        envelopes[video] = envelope

    def read_video(
        source: Path, *, on_progress: Callable[[int], None] | None = None
    ) -> Generator[bytes, None, None]:
        envelope = envelopes[source]
        yield from encode_packets(envelope, prepare_manifest(envelope))

    monkeypatch.setattr(cli_module, "read_video", read_video)
    failed_output = output / "b.bin"
    if overwrite:
        failed_output.write_bytes(b"keep")
    assert cli_module.main(
        [
            "decode",
            "--input-dir",
            str(videos),
            "--out-dir",
            str(output),
            "--key",
            "correct",
            "--max-bytes",
            "512",
        ]
        + (["--overwrite"] if overwrite else [])
    ) == (0 if failure == "none" else 1)
    assert (output / "a.bin").read_bytes() == b"payload" * 40
    assert (output / "c.bin").read_bytes() == b"payload" * 40
    assert not list(output.glob(".qr-video-*.part"))
    captured = capsys.readouterr()
    if failure == "none":
        assert failed_output.read_bytes() == b"payload" * 40
        assert "3 succeeded, 0 failed, 0 skipped" in captured.out
    else:
        assert failed_output.read_bytes() == b"keep" if overwrite else not failed_output.exists()
        assert "2 succeeded, 1 failed, 0 skipped" in captured.out
        assert str(videos / "b.bin.mp4") in captured.err.split("Failed files:")[1]


@pytest.mark.parametrize("overwrite", [False, True])
def test_batch_interrupt_stops_and_cleans_pending_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    overwrite: bool,
) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / name).write_bytes(b"payload")
    output = tmp_path / "output"
    output.mkdir()
    if overwrite:
        (output / "b.mp4").write_bytes(b"keep")
    workspaces: list[Path] = []

    def interrupt(
        packets: Iterable[bytes],
        destination: Path,
        *,
        profile: Profile,
        crf: int,
        on_progress: object,
    ) -> int:
        workspaces.append(destination.parent)
        if len(workspaces) == 2:
            destination.write_bytes(b"partial")
            raise KeyboardInterrupt
        count = sum(1 for _ in packets)
        destination.write_bytes(b"complete")
        return (count + 1) // 2

    monkeypatch.setattr(cli_module, "write_video", interrupt)
    assert (
        cli_module.main(
            ["encode", "--input-dir", str(tmp_path), "--out-dir", str(output)]
            + (["--overwrite"] if overwrite else [])
        )
        == 130
    )
    assert len(workspaces) == 2
    assert all(not workspace.exists() for workspace in workspaces)
    assert (output / "a.mp4").read_bytes() == b"complete"
    if overwrite:
        assert (output / "b.mp4").read_bytes() == b"keep"
    else:
        assert not (output / "b.mp4").exists()
    assert not (output / "c.mp4").exists()
    assert not list(output.glob(".qr-video-*.part"))
    assert "interrupted" in capsys.readouterr().err
