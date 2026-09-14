import hashlib
import io
import os
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import BinaryIO

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from qr_video import envelope
from qr_video.envelope import (
    AAD_DOMAIN,
    CHUNK_SIZE,
    DATA_RECORD,
    DEFAULT_MAX_BYTES,
    FINAL_RECORD,
    FINAL_STRUCT,
    FLAG_COMPRESSED,
    FLAG_ENCRYPTED,
    HEADER_STRUCT,
    KDF_INFO,
    MAGIC,
    MAX_DATA_RECORDS,
    MAX_ENVELOPE_SIZE,
    MAX_FILE_SIZE,
    NONCE_SIZE,
    RECORD_STRUCT,
    TAG_SIZE,
    VERSION,
    decode_envelope,
    encode_envelope,
)
from qr_video.errors import QRVideoError

KEY = bytes(range(32))


def _reference_nonce(salt: bytes, index: int) -> bytes:
    counter = index.to_bytes(NONCE_SIZE, "big")
    return bytes(base ^ count for base, count in zip(salt[:NONCE_SIZE], counter, strict=True))


def _records(blob: bytes) -> list[bytes]:
    result = []
    offset = HEADER_STRUCT.size
    while offset < len(blob):
        _, _, length = RECORD_STRUCT.unpack_from(blob, offset)
        end = offset + RECORD_STRUCT.size + length
        result.append(blob[offset:end])
        offset = end
    assert offset == len(blob)
    return result


def _crafted_envelope(
    original: bytes,
    payload: bytes,
    *,
    compressed: bool = True,
    key: bytes | None = None,
    declared_size: int | None = None,
    pieces: list[bytes] | None = None,
) -> bytes:
    salt = b"0123456789abcdef" * 2
    flags = (FLAG_COMPRESSED if compressed else 0) | (FLAG_ENCRYPTED if key else 0)
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        0,
        CHUNK_SIZE,
        len(original) if declared_size is None else declared_size,
        salt,
    )
    cipher = (
        AESGCM(HKDF(hashes.SHA256(), 32, salt, KDF_INFO).derive(key)) if key is not None else None
    )

    def record(kind: int, index: int, data: bytes) -> bytes:
        prefix = RECORD_STRUCT.pack(kind, index, len(data) + (TAG_SIZE if cipher else 0))
        if cipher is not None:
            data = cipher.encrypt(_reference_nonce(salt, index), data, AAD_DOMAIN + header + prefix)
        return prefix + data

    if pieces is None:
        pieces = [
            payload[start : start + CHUNK_SIZE] for start in range(0, len(payload), CHUNK_SIZE)
        ]
    data_records = b"".join(record(DATA_RECORD, index, data) for index, data in enumerate(pieces))
    final = FINAL_STRUCT.pack(
        len(original),
        len(payload),
        len(pieces),
        hashlib.sha256(original).digest(),
        hashlib.sha256(header + data_records).digest(),
    )
    return header + data_records + record(FINAL_RECORD, len(pieces), final)


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("size", [0, 1, 256, CHUNK_SIZE, CHUNK_SIZE + 19, 2 * CHUNK_SIZE + 71])
def test_round_trip(tmp_path: Path, encrypted: bool, compressed: bool, size: int) -> None:
    data = os.urandom(size)
    source, stored, recovered = tmp_path / "source", tmp_path / "stored", tmp_path / "recovered"
    source.write_bytes(data)
    key = KEY if encrypted else None
    encoded = encode_envelope(source, stored, key=key, compress=compressed)
    decoded = decode_envelope(stored, recovered, key=key)
    assert recovered.read_bytes() == data
    assert encoded == decoded
    assert encoded.original_size == size
    assert encoded.stored_size == stored.stat().st_size
    assert encoded.encrypted is encrypted
    assert encoded.compressed is compressed
    assert encoded.sha256 == hashlib.sha256(data).hexdigest()
    assert set(tmp_path.iterdir()) == {source, stored, recovered}
    for record in _records(stored.read_bytes()):
        assert len(record) - RECORD_STRUCT.size <= 1024 * 1024
    if os.name == "posix":
        assert recovered.stat().st_mode & 0o777 == 0o600


def test_info_is_frozen(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"data")
    info = encode_envelope(source, tmp_path / "stored")
    with pytest.raises(FrozenInstanceError):
        info.original_size = 100  # type: ignore[misc]


@pytest.mark.parametrize("size", [0, 32, 2 * CHUNK_SIZE + 19])
def test_fresh_salts_keys_nonces_and_ciphertext(tmp_path: Path, size: int) -> None:
    source = tmp_path / "source"
    source.write_bytes(os.urandom(size))
    first, second = tmp_path / "first", tmp_path / "second"
    encode_envelope(source, first, key=KEY)
    encode_envelope(source, second, key=KEY)
    first_blob, second_blob = first.read_bytes(), second.read_bytes()
    first_header = first_blob[: HEADER_STRUCT.size]
    second_header = second_blob[: HEADER_STRUCT.size]
    first_salt = HEADER_STRUCT.unpack(first_header)[-1]
    second_salt = HEADER_STRUCT.unpack(second_header)[-1]
    assert first_salt != second_salt
    assert _records(first_blob) != _records(second_blob)
    file_nonces = []
    derived_keys = []
    for blob in (first_blob, second_blob):
        header = blob[: HEADER_STRUCT.size]
        salt = HEADER_STRUCT.unpack(header)[-1]
        derived = HKDF(hashes.SHA256(), 32, salt, KDF_INFO).derive(KEY)
        derived_keys.append(derived)
        cipher = AESGCM(derived)
        plaintext = []
        indices = []
        nonces = []
        for record in _records(blob):
            prefix = record[: RECORD_STRUCT.size]
            kind, index, length = RECORD_STRUCT.unpack(prefix)
            assert length == len(record) - RECORD_STRUCT.size
            indices.append(index)
            nonce = _reference_nonce(salt, index)
            nonces.append(nonce)
            data = cipher.decrypt(nonce, record[RECORD_STRUCT.size :], AAD_DOMAIN + header + prefix)
            if kind == DATA_RECORD:
                plaintext.append(data)
            else:
                assert kind == FINAL_RECORD
                assert index == len(plaintext)
        assert indices == list(range(len(indices)))
        assert len(set(nonces)) == len(nonces)
        assert nonces[0] == salt[:NONCE_SIZE]
        assert all(len(nonce) == NONCE_SIZE for nonce in nonces)
        assert b"".join(plaintext) == source.read_bytes()
        file_nonces.append(set(nonces))
    assert derived_keys[0] != derived_keys[1]
    assert file_nonces[0].isdisjoint(file_nonces[1])


def test_nonce_mapping_includes_a_unique_final_record_at_the_size_limit() -> None:
    salt = bytes(range(32))
    assert envelope._record_nonce(salt, 0) == bytes.fromhex("000102030405060708090a0b")
    assert envelope._record_nonce(salt, 1) == bytes.fromhex("000102030405060708090a0a")
    assert envelope._record_nonce(salt, 256) == bytes.fromhex("000102030405060708090b0b")
    nonces = {envelope._record_nonce(salt, index) for index in range(MAX_DATA_RECORDS + 1)}
    assert len(nonces) == MAX_DATA_RECORDS + 1


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("wrong_key", [None, b"z" * 32])
@pytest.mark.parametrize("plaintext", [b"", b"secret" * 5000])
def test_missing_or_wrong_key_never_publishes(
    tmp_path: Path, compress: bool, wrong_key: bytes | None, plaintext: bytes
) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(plaintext)
    encode_envelope(source, stored, key=KEY, compress=compress)
    with pytest.raises(QRVideoError, match="requires|authentication"):
        decode_envelope(stored, output, key=wrong_key)
    assert set(tmp_path.iterdir()) == {source, stored}


@pytest.mark.parametrize("bad_key", [b"", b"x" * 31, b"x" * 33, "x" * 32, bytearray(32)])
def test_malformed_keys_fail_before_output(tmp_path: Path, bad_key: bytes) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"secret")
    with pytest.raises(QRVideoError, match="exactly 32"):
        encode_envelope(source, output, key=bad_key)
    encode_envelope(source, stored, key=KEY)
    with pytest.raises(QRVideoError, match="exactly 32"):
        decode_envelope(stored, output, key=bad_key)
    assert set(tmp_path.iterdir()) == {source, stored}


def test_key_is_not_ignored_for_plaintext(tmp_path: Path) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"data")
    encode_envelope(source, stored)
    with pytest.raises(QRVideoError, match="unencrypted"):
        decode_envelope(stored, output, key=KEY)
    assert not output.exists()


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize(
    "offset",
    [
        0,
        8,
        9,
        10,
        12,
        23,
        24,
        HEADER_STRUCT.size,
        HEADER_STRUCT.size + 8,
        HEADER_STRUCT.size + 12,
        HEADER_STRUCT.size + RECORD_STRUCT.size,
        -1,
    ],
)
def test_metadata_and_data_tampering_preserves_existing_output(
    tmp_path: Path, encrypted: bool, offset: int
) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"original bytes" * 100)
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key)
    blob = bytearray(stored.read_bytes())
    blob[offset] ^= 1
    stored.write_bytes(blob)
    output.write_bytes(b"preserve me")
    with pytest.raises(QRVideoError):
        decode_envelope(stored, output, key=key, overwrite=True)
    assert output.read_bytes() == b"preserve me"
    assert set(tmp_path.iterdir()) == {source, stored, output}


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize(
    "mutation", ["delete", "swap", "duplicate", "renumber", "no-final", "extra"]
)
def test_record_order_and_completeness(tmp_path: Path, encrypted: bool, mutation: str) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(os.urandom(2 * CHUNK_SIZE + 23))
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key)
    blob = stored.read_bytes()
    records = _records(blob)
    assert len(records) == 4
    if mutation == "delete":
        del records[1]
    elif mutation == "swap":
        records[0], records[1] = records[1], records[0]
    elif mutation == "duplicate":
        records.insert(1, records[0])
    elif mutation == "renumber":
        records[0], records[1] = records[1], records[0]
        for index in range(2):
            kind, _, length = RECORD_STRUCT.unpack_from(records[index])
            records[index] = (
                RECORD_STRUCT.pack(kind, index, length) + records[index][RECORD_STRUCT.size :]
            )
    elif mutation == "no-final":
        records.pop()
    else:
        records.append(b"extra")
    stored.write_bytes(blob[: HEADER_STRUCT.size] + b"".join(records))
    with pytest.raises(QRVideoError):
        decode_envelope(stored, output, key=key)
    assert set(tmp_path.iterdir()) == {source, stored}


@pytest.mark.parametrize("encrypted", [False, True])
def test_every_truncation_boundary_fails(tmp_path: Path, encrypted: bool) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"binary\0\xff")
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key, compress=True)
    original = stored.read_bytes()
    for boundary in range(len(original)):
        stored.write_bytes(original[:boundary])
        with pytest.raises(QRVideoError):
            decode_envelope(stored, output, key=key)
        assert not output.exists()
        assert set(tmp_path.iterdir()) == {source, stored}


@pytest.mark.parametrize("encrypted", [False, True])
def test_forged_wire_length_is_bounded(tmp_path: Path, encrypted: bool) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"data")
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key)
    blob = stored.read_bytes()
    stored.write_bytes(
        blob[: HEADER_STRUCT.size]
        + RECORD_STRUCT.pack(DATA_RECORD, 0, 0xFFFFFFFF)
        + blob[HEADER_STRUCT.size + RECORD_STRUCT.size :]
    )
    with pytest.raises(QRVideoError, match="length"):
        decode_envelope(stored, output, key=key)
    assert not output.exists()


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("field_offset", [7, 15, 23, 24, 56])
def test_final_fields_are_validated_even_with_valid_authentication(
    tmp_path: Path, encrypted: bool, field_offset: int
) -> None:
    original = b"verified content"
    key = KEY if encrypted else None
    blob = _crafted_envelope(original, zlib.compress(original), key=key)
    header = blob[: HEADER_STRUCT.size]
    records = _records(blob)
    prefix = records[-1][: RECORD_STRUCT.size]
    _, index, _ = RECORD_STRUCT.unpack(prefix)
    final = records[-1][RECORD_STRUCT.size :]
    salt = HEADER_STRUCT.unpack(header)[-1]
    nonce = _reference_nonce(salt, index)
    aad = AAD_DOMAIN + header + prefix
    cipher = None
    if key is not None:
        cipher = AESGCM(HKDF(hashes.SHA256(), 32, salt, KDF_INFO).derive(key))
        final = cipher.decrypt(nonce, final, aad)
    modified = bytearray(final)
    modified[field_offset] ^= 1
    final = bytes(modified)
    if cipher is not None:
        final = cipher.encrypt(nonce, final, aad)
    records[-1] = prefix + final
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(header + b"".join(records))
    with pytest.raises(QRVideoError, match="final size, count, or checksum"):
        decode_envelope(stored, output, key=key)
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("invalid", ["truncated", "trailing", "concatenated", "raw", "missing"])
def test_invalid_compressed_streams(tmp_path: Path, encrypted: bool, invalid: str) -> None:
    original = b"compressed content" * 100
    payload = zlib.compress(original)
    if invalid == "truncated":
        payload = payload[:-1]
    elif invalid == "trailing":
        payload += b"trailing"
    elif invalid == "concatenated":
        payload += zlib.compress(b"")
    elif invalid == "raw":
        payload = original
    else:
        payload = b""
    key = KEY if encrypted else None
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(_crafted_envelope(original, payload, key=key))
    with pytest.raises(QRVideoError):
        decode_envelope(stored, output, key=key)
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("encrypted", [False, True])
def test_compression_bomb_cannot_write_past_declared_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encrypted: bool
) -> None:
    original = b"x" * (4 * CHUNK_SIZE)
    declared_size = 8192
    key = KEY if encrypted else None
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(
        _crafted_envelope(original, zlib.compress(original), declared_size=declared_size, key=key)
    )
    plaintext = io.BytesIO()

    @contextmanager
    def recording_output(path: Path, *, overwrite: bool = False) -> Iterator[BinaryIO]:
        yield plaintext

    monkeypatch.setattr(envelope, "atomic_output", recording_output)
    with pytest.raises(QRVideoError, match="exceeds the declared"):
        decode_envelope(stored, output, key=key, max_bytes=declared_size)
    assert len(plaintext.getvalue()) <= declared_size
    assert not output.exists()


@pytest.mark.parametrize("encrypted", [False, True])
def test_compressed_long_repeated_runs(tmp_path: Path, encrypted: bool) -> None:
    original = b"\0" * (5 * CHUNK_SIZE + 27)
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(original)
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key, compress=True)
    assert stored.stat().st_size < len(original) // 100
    decode_envelope(stored, output, key=key)
    assert output.read_bytes() == original


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("pieces", [[b"", b"abc"], [b"a", b"bc"]])
def test_noncanonical_data_records_are_rejected(
    tmp_path: Path, encrypted: bool, pieces: list[bytes]
) -> None:
    stored, output = tmp_path / "stored", tmp_path / "output"
    key = KEY if encrypted else None
    stored.write_bytes(_crafted_envelope(b"abc", b"abc", compressed=False, key=key, pieces=pieces))
    with pytest.raises(QRVideoError, match="length"):
        decode_envelope(stored, output, key=key)
    assert list(tmp_path.iterdir()) == [stored]


def test_limits_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"data")
    stored.write_bytes(_crafted_envelope(b"", b"", declared_size=MAX_FILE_SIZE + 1))
    with pytest.raises(QRVideoError, match="header"):
        decode_envelope(stored, output)
    monkeypatch.setattr(envelope, "MAX_FILE_SIZE", 3)
    with pytest.raises(QRVideoError, match="limit"):
        encode_envelope(source, output)
    assert set(tmp_path.iterdir()) == {source, stored}


@pytest.mark.parametrize("max_bytes", [0, -1, MAX_FILE_SIZE + 1, True, False, 1.0, "1024", None])
def test_invalid_output_ceiling_is_rejected(tmp_path: Path, max_bytes: int) -> None:
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(_crafted_envelope(b"", zlib.compress(b"")))
    with pytest.raises(QRVideoError, match="max_bytes must be an integer"):
        decode_envelope(stored, output, max_bytes=max_bytes)
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_custom_output_ceiling(tmp_path: Path, encrypted: bool, compressed: bool) -> None:
    original = b"x" * 4096
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(original)
    output.write_bytes(b"preserve me")
    key = KEY if encrypted else None
    encode_envelope(source, stored, key=key, compress=compressed)
    with pytest.raises(QRVideoError, match="4095-byte output limit"):
        decode_envelope(stored, output, key=key, overwrite=True, max_bytes=len(original) - 1)
    assert output.read_bytes() == b"preserve me"
    info = decode_envelope(stored, output, key=key, overwrite=True, max_bytes=len(original))
    assert info.original_size == len(original)
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source, stored, output}


@pytest.mark.parametrize("original", [b"", b"x"])
@pytest.mark.parametrize("max_bytes", [1, MAX_FILE_SIZE])
def test_output_ceiling_endpoints(tmp_path: Path, original: bytes, max_bytes: int) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(original)
    encode_envelope(source, stored, key=KEY, compress=True)
    decode_envelope(stored, output, key=KEY, max_bytes=max_bytes)
    assert output.read_bytes() == original


def test_default_ceiling_rejects_header_before_record_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(
        _crafted_envelope(b"", b"", declared_size=DEFAULT_MAX_BYTES + 1)[: HEADER_STRUCT.size]
    )
    sizes = []
    original_read = envelope._read_exact

    def recording_read(stream: BinaryIO, size: int) -> bytes:
        sizes.append(size)
        return original_read(stream, size)

    def unexpected_work(*args: object, **kwargs: object) -> None:
        pytest.fail("The decoder processed records or created output before checking max_bytes.")

    monkeypatch.setattr(envelope, "_read_exact", recording_read)
    monkeypatch.setattr(envelope, "_cipher", unexpected_work)
    monkeypatch.setattr(envelope, "atomic_output", unexpected_work)
    with pytest.raises(QRVideoError, match="1073741824-byte output limit"):
        decode_envelope(stored, output)
    assert sizes == [HEADER_STRUCT.size]
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("declared_size", [DEFAULT_MAX_BYTES + 1, MAX_FILE_SIZE])
def test_raised_decode_ceiling_accepts_large_header(tmp_path: Path, declared_size: int) -> None:
    stored, output = tmp_path / "stored", tmp_path / "output"
    stored.write_bytes(
        _crafted_envelope(b"", b"", declared_size=declared_size)[: HEADER_STRUCT.size]
    )
    with pytest.raises(QRVideoError, match="Truncated envelope"):
        decode_envelope(stored, output, max_bytes=declared_size)
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("declared_size", [DEFAULT_MAX_BYTES + 1, MAX_FILE_SIZE])
def test_large_encoding_sizes_pass_header_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared_size: int
) -> None:
    source, stored = tmp_path / "source", tmp_path / "stored"
    with source.open("wb") as stream:
        stream.truncate(declared_size)

    def stop_before_writing(*args: object, **kwargs: object) -> None:
        raise RuntimeError("size accepted, stop before processing sparse data")

    monkeypatch.setattr(envelope, "atomic_output", stop_before_writing)
    with pytest.raises(RuntimeError, match="size accepted"):
        encode_envelope(source, stored)
    assert list(tmp_path.iterdir()) == [source]


def test_maximum_envelope_size_checked_before_reading(tmp_path: Path) -> None:
    stored, output = tmp_path / "stored", tmp_path / "output"
    with stored.open("wb") as stream:
        stream.truncate(MAX_ENVELOPE_SIZE + 1)
    with pytest.raises(QRVideoError, match="maximum stored size"):
        decode_envelope(stored, output)
    assert list(tmp_path.iterdir()) == [stored]


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_source_output_aliases_are_rejected(tmp_path: Path, alias: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"unchanged")
    output = source if alias == "same" else tmp_path / "output"
    if alias == "hardlink":
        os.link(source, output)
    elif alias == "symlink":
        output.symlink_to(source)
    with pytest.raises(QRVideoError, match="protected|symlink"):
        encode_envelope(source, output)
    with pytest.raises(QRVideoError, match="protected|symlink"):
        decode_envelope(source, output, overwrite=True)
    assert source.read_bytes() == b"unchanged"


def test_decode_overwrite_requires_opt_in(tmp_path: Path) -> None:
    source, stored, output = tmp_path / "source", tmp_path / "stored", tmp_path / "output"
    source.write_bytes(b"new")
    output.write_bytes(b"old")
    encode_envelope(source, stored, key=KEY)
    with pytest.raises(QRVideoError, match="already exists"):
        decode_envelope(stored, output, key=KEY)
    assert output.read_bytes() == b"old"
    decode_envelope(stored, output, key=KEY, overwrite=True)
    assert output.read_bytes() == b"new"


def test_encode_failure_cleans_private_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, stored = tmp_path / "source", tmp_path / "stored"
    source.write_bytes(b"source")

    def fail(*args: object, **kwargs: object) -> bytes:
        raise QRVideoError("injected encoding failure")

    monkeypatch.setattr(envelope, "_record_bytes", fail)
    with pytest.raises(QRVideoError, match="injected"):
        encode_envelope(source, stored, key=KEY)
    assert list(tmp_path.iterdir()) == [source]
