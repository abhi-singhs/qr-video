import os
import stat
from pathlib import Path

import pytest

from qr_video import safeio
from qr_video.errors import QRVideoError
from qr_video.safeio import atomic_output, check_output, generate_key, read_key


def test_generated_keys_are_complete_and_distinct(tmp_path: Path) -> None:
    first, second = tmp_path / "first.key", tmp_path / "second.key"
    generate_key(first)
    generate_key(second)
    assert len(read_key(first)) == 32
    assert read_key(first) != read_key(second)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.key", "second.key"]
    if os.name == "posix":
        assert stat.S_IMODE(first.stat().st_mode) == 0o600


def test_key_generation_never_overwrites(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_bytes(b"existing")
    with pytest.raises(QRVideoError, match="already exists"):
        generate_key(key)
    assert key.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [key]


@pytest.mark.parametrize("size", [0, 1, 31, 33, 64, 1024 * 1024])
def test_invalid_key_sizes(tmp_path: Path, size: int) -> None:
    key = tmp_path / "key"
    key.write_bytes(b"x" * size)
    with pytest.raises(QRVideoError, match="exactly 32 raw bytes"):
        read_key(key)


def test_unreadable_key_paths(tmp_path: Path) -> None:
    with pytest.raises(QRVideoError, match="Cannot read"):
        read_key(tmp_path / "missing")
    with pytest.raises(QRVideoError):
        read_key(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO")
def test_key_fifo_does_not_block(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(QRVideoError, match="not a regular file"):
        read_key(fifo)


def test_check_output_and_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "output"
    check_output(output)
    output.write_bytes(b"old")
    with pytest.raises(QRVideoError, match="already exists"):
        check_output(output)
    check_output(output, overwrite=True)
    with pytest.raises(QRVideoError, match="not a regular file"):
        check_output(tmp_path, overwrite=True)
    with pytest.raises(QRVideoError, match="directory does not exist"):
        check_output(tmp_path / "missing" / "output")


@pytest.mark.parametrize("alias", ["same", "relative", "hardlink", "symlink"])
def test_protected_aliases(tmp_path: Path, alias: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    if alias == "same":
        output = source
    elif alias == "relative":
        output = tmp_path / ".." / tmp_path.name / source.name
    elif alias == "hardlink":
        os.link(source, output)
    else:
        output.symlink_to(source)
    with pytest.raises(QRVideoError, match="protected|symlink"):
        check_output(output, overwrite=True, protected=(source,))
    assert source.read_bytes() == b"source"


def test_nonexistent_protected_path(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(QRVideoError, match="protected"):
        check_output(output, protected=(output,))


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("aliased_parent", [False, True])
def test_symlink_outputs_are_rejected(
    tmp_path: Path, dangling: bool, overwrite: bool, aliased_parent: bool
) -> None:
    target = tmp_path / "target"
    if not dangling:
        target.write_bytes(b"target")
    parent = tmp_path
    if aliased_parent:
        real = tmp_path / "real"
        real.mkdir()
        parent = tmp_path / "alias"
        parent.symlink_to(real, target_is_directory=True)
    output = parent / "output"
    output.symlink_to(target)
    with pytest.raises(QRVideoError, match="symlink"), atomic_output(output, overwrite=overwrite):
        pytest.fail("Symlink output was opened.")
    assert output.is_symlink()
    if not dangling:
        assert target.read_bytes() == b"target"


def test_key_generation_allows_symlink_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    generate_key(alias / "key")
    assert len(read_key(alias / "key")) == 32
    assert read_key(real / "key") == read_key(alias / "key")
    assert list(real.iterdir()) == [real / "key"]
    if os.name == "posix":
        assert stat.S_IMODE((real / "key").stat().st_mode) == 0o600


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_output_allows_symlink_parent(tmp_path: Path, overwrite: bool) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "output"
    if overwrite:
        output.write_bytes(b"old")
    with atomic_output(output, overwrite=overwrite) as stream:
        stream.write(b"new")
        stream.flush()
        assert len(list(real.glob(".qr-video-*.part"))) == 1
        assert output.read_bytes() == b"old" if overwrite else not output.exists()
    assert output.read_bytes() == b"new"
    assert (real / "output").read_bytes() == b"new"
    assert list(real.iterdir()) == [real / "output"]
    with pytest.raises(QRVideoError, match="already exists"), atomic_output(output):
        pytest.fail("An existing file was opened without overwrite.")


@pytest.mark.parametrize("hardlink", [False, True])
@pytest.mark.parametrize("protected_alias", [False, True])
def test_symlink_parent_does_not_bypass_protected_files(
    tmp_path: Path, hardlink: bool, protected_alias: bool
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    source = real / "source"
    source.write_bytes(b"protected")
    output = alias / "source"
    if hardlink:
        output = alias / "hardlink"
        os.link(source, output)
    protected = alias / "source" if protected_alias else source
    with pytest.raises(QRVideoError, match="protected"):
        check_output(output, overwrite=True, protected=(protected,))
    assert source.read_bytes() == b"protected"


def test_atomic_output_keeps_the_original_resolved_parent(tmp_path: Path) -> None:
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    with atomic_output(alias / "output") as stream:
        stream.write(b"complete")
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
    assert (original / "output").read_bytes() == b"complete"
    assert list(original.iterdir()) == [original / "output"]
    assert list(replacement.iterdir()) == []


def test_failed_output_with_symlink_parent_is_cleaned(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="injected"), atomic_output(alias / "output") as stream:
        stream.write(b"partial")
        raise RuntimeError("injected")
    assert list(real.iterdir()) == []


def test_atomic_output_is_private_until_complete(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with atomic_output(output) as stream:
        stream.write(b"complete")
        stream.flush()
        assert not output.exists()
        pending = list(tmp_path.glob(".qr-video-*.part"))
        assert len(pending) == 1
        if os.name == "posix":
            assert stat.S_IMODE(pending[0].stat().st_mode) == 0o600
    assert output.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [output]


def test_atomic_overwrite_retains_old_file_until_publish(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.write_bytes(b"old")
    with atomic_output(output, overwrite=True) as stream:
        stream.write(b"new")
        assert output.read_bytes() == b"old"
    assert output.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_body_cleans_private_output(tmp_path: Path, overwrite: bool) -> None:
    output = tmp_path / "output"
    if overwrite:
        output.write_bytes(b"old")
    unrelated = tmp_path / ".qr-video-unrelated.part"
    unrelated.write_bytes(b"leave me")
    with (
        pytest.raises(RuntimeError, match="test failure"),
        atomic_output(output, overwrite=overwrite) as stream,
    ):
        stream.write(b"partial plaintext")
        raise RuntimeError("test failure")
    assert unrelated.read_bytes() == b"leave me"
    assert output.read_bytes() == b"old" if overwrite else not output.exists()
    assert set(tmp_path.iterdir()) == ({output, unrelated} if overwrite else {unrelated})


@pytest.mark.parametrize("operation", ["fsync", "link", "replace"])
def test_failed_publication_cleans_private_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    output = tmp_path / "output"
    overwrite = operation == "replace"
    if overwrite:
        output.write_bytes(b"old")

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(safeio.os, operation, fail)
    with (
        pytest.raises(QRVideoError, match="injected write failure"),
        atomic_output(output, overwrite=overwrite) as stream,
    ):
        stream.write(b"new")
    assert output.read_bytes() == b"old" if overwrite else not output.exists()
    assert list(tmp_path.iterdir()) == ([output] if overwrite else [])


def test_failed_key_fsync_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(fd: int) -> None:
        raise OSError("fsync failure")

    monkeypatch.setattr(safeio.os, "fsync", fail)
    with pytest.raises(QRVideoError, match="fsync failure"):
        generate_key(tmp_path / "key")
    assert list(tmp_path.iterdir()) == []


def test_racing_no_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    original_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"racing writer")
        original_link(source, destination)

    monkeypatch.setattr(safeio.os, "link", racing_link)
    with pytest.raises(QRVideoError), atomic_output(output) as stream:
        stream.write(b"our data")
    assert output.read_bytes() == b"racing writer"
    assert list(tmp_path.iterdir()) == [output]


def test_symlink_created_before_publish_is_rejected(tmp_path: Path) -> None:
    output, target = tmp_path / "output", tmp_path / "target"
    target.write_bytes(b"target")
    with (
        pytest.raises(QRVideoError, match="symlink"),
        atomic_output(output, overwrite=True) as stream,
    ):
        stream.write(b"new")
        output.symlink_to(target)
    assert output.is_symlink()
    assert target.read_bytes() == b"target"
    assert set(tmp_path.iterdir()) == {output, target}
