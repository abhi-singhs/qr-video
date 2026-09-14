import hashlib
import os
import random
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from qr_video.errors import QRVideoError
from qr_video.profiles import CONSERVATIVE
from qr_video.transport import (
    DATA,
    MANIFEST,
    MAX_STREAM_SIZE,
    PACKET_OVERHEAD,
    SHARD_SIZE,
    InvalidPacket,
    Manifest,
    Packet,
    encode_packets,
    prepare_manifest,
    recover_stream,
    transport_stats,
)


def packets_for(
    tmp_path: Path,
    data: bytes,
    *,
    k: int = 100,
    m: int = 25,
    interleave: int = 4,
) -> tuple[Manifest, list[bytes]]:
    source = tmp_path / "source"
    source.write_bytes(data)
    manifest = prepare_manifest(source, k=k, m=m, interleave=interleave)
    return manifest, list(encode_packets(source, manifest))


def decode_packets(
    tmp_path: Path,
    packets: list[bytes],
    expected: bytes,
) -> None:
    output = tmp_path / "recovered"
    manifest, stats = recover_stream(packets, output)
    assert manifest.stream_sha256 == hashlib.sha256(expected).digest()
    assert output.read_bytes() == expected
    assert stats.blocks == manifest.blocks


def test_raw_binary_packet_and_exact_capacity() -> None:
    payload = (bytes(range(256)) * 2)[:SHARD_SIZE]
    packet = Packet(DATA, os.urandom(16), 17, 124, 100, 25, SHARD_SIZE, payload)
    raw = packet.to_bytes()
    assert PACKET_OVERHEAD == 40
    assert SHARD_SIZE == 342
    assert len(raw) == CONSERVATIVE.qr_capacity == 382
    assert Packet.from_bytes(raw) == packet
    with pytest.raises(InvalidPacket, match="payload size"):
        replace(packet, payload=payload + b"\x00").to_bytes()
    with pytest.raises(InvalidPacket, match="CRC"):
        Packet.from_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))


@pytest.mark.parametrize(
    ("k", "m", "interleave"),
    [(0, 1, 4), (100, 0, 4), (100, 101, 4), (200, 100, 4), (100, 25, 0), (100, 25, 17)],
)
def test_invalid_coding(k: int, m: int, interleave: int) -> None:
    manifest = Manifest(os.urandom(16), 1, os.urandom(32), k, m, interleave)
    with pytest.raises(QRVideoError):
        manifest.packet()


@pytest.mark.parametrize("size", [1, 341, 342, 343, 34200, 34201, 34200 * 5 + 219])
def test_transport_boundaries_and_padding(tmp_path: Path, size: int) -> None:
    data = os.urandom(size)
    manifest, packets = packets_for(tmp_path, data)
    stats = transport_stats(manifest)
    assert stats["qr_packets"] == len(packets)
    assert stats["images"] == (len(packets) + 1) // 2
    assert all(len(packet) <= 382 for packet in packets)
    assert Manifest.from_packet(Packet.from_bytes(packets[0])) == manifest
    decode_packets(tmp_path, packets, data)


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_exact_outer_erasure_threshold(tmp_path: Path, seed: int) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    manifest, packets = packets_for(tmp_path, data)
    shards = [packet for packet in packets if Packet.from_bytes(packet).kind == DATA]
    assert len(shards) == 125
    random.Random(seed).shuffle(shards)
    # All recovery happens in zfec, with no QR decoder involved in this threshold test.
    survivors = shards[25:]
    assert len(survivors) == 100
    decode_packets(tmp_path, survivors + [manifest.packet()], data)
    failed_output = tmp_path / "insufficient"
    with pytest.raises(QRVideoError, match="need 100, got 99"):
        recover_stream(survivors[1:] + [manifest.packet()], failed_output)
    assert not failed_output.exists()


def test_parity_only_small_block_and_shortening(tmp_path: Path) -> None:
    data = os.urandom(342 + 1)
    manifest, packets = packets_for(tmp_path, data)
    assert manifest.block_parameters(0) == (2, 1, 343)
    survivors = [
        raw
        for raw in packets
        if Packet.from_bytes(raw).kind == MANIFEST or Packet.from_bytes(raw).index != 0
    ]
    decode_packets(tmp_path, survivors, data)


def test_repeated_bootstrap_survives_missing_leading_frames(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    _, packets = packets_for(tmp_path, data)
    # Lose both leading manifests and the first 20 shards. Later copies bootstrap recovery.
    remaining = packets[22:]
    assert sum(Packet.from_bytes(raw).kind == DATA for raw in remaining) == 105
    decode_packets(tmp_path, remaining, data)


def test_multiblock_interleaving_duplicates_and_reordering(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE * 5 + 12)
    manifest, packets = packets_for(tmp_path, data)
    shards = [Packet.from_bytes(raw) for raw in packets if Packet.from_bytes(raw).kind == DATA]
    assert [(p.block, p.index) for p in shards[:8]] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    # A burst of 100 data packets erases 25 shards in each of the first four blocks.
    survivors = [p.to_bytes() for p in shards[100:]]
    survivors += survivors[::7]
    random.Random(13).shuffle(survivors)
    survivors.append(manifest.packet())
    decode_packets(tmp_path, survivors, data)


def test_crc_failures_are_erasures(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    _, packets = packets_for(tmp_path, data)
    damaged = 0
    for index, raw in enumerate(packets):
        if Packet.from_bytes(raw).kind == DATA and damaged < 25:
            packets[index] = raw[:-1] + bytes([raw[-1] ^ 1])
            damaged += 1
    output = tmp_path / "recovered"
    _, stats = recover_stream(packets, output)
    assert stats.invalid_packets == 25
    assert output.read_bytes() == data


def test_conflicting_copies_are_permanent_erasures(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    _, packets = packets_for(tmp_path, data)
    original = next(raw for raw in packets if Packet.from_bytes(raw).kind == DATA)
    parsed = Packet.from_bytes(original)
    conflict = replace(parsed, payload=os.urandom(SHARD_SIZE)).to_bytes()
    output = tmp_path / "recovered"
    _, stats = recover_stream(packets + [conflict, original, original], output)
    assert stats.conflicts == 1
    assert output.read_bytes() == data


def test_conflict_can_exhaust_recovery_budget(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    manifest, packets = packets_for(tmp_path, data)
    originals = [raw for raw in packets if Packet.from_bytes(raw).kind == DATA][:100]
    conflict = replace(Packet.from_bytes(originals[0]), payload=os.urandom(SHARD_SIZE)).to_bytes()
    output = tmp_path / "failed"
    with pytest.raises(QRVideoError, match="need 100, got 99"):
        recover_stream([manifest.packet(), *originals, conflict], output)
    assert not output.exists()


def test_corrupt_stream_never_retains_output(tmp_path: Path) -> None:
    data = os.urandom(100 * SHARD_SIZE)
    manifest, packets = packets_for(tmp_path, data)
    originals = [raw for raw in packets if Packet.from_bytes(raw).kind == DATA][:100]
    originals[0] = replace(
        Packet.from_bytes(originals[0]),
        payload=os.urandom(SHARD_SIZE),
    ).to_bytes()
    output = tmp_path / "failed"
    with pytest.raises(QRVideoError, match="SHA-256"):
        recover_stream([manifest.packet(), *originals], output)
    assert not output.exists()


def test_missing_or_conflicting_manifest(tmp_path: Path) -> None:
    manifest, packets = packets_for(tmp_path, b"test")
    shards = [raw for raw in packets if Packet.from_bytes(raw).kind == DATA]
    output = tmp_path / "failed"
    with pytest.raises(QRVideoError, match="No valid manifest"):
        recover_stream(shards, output)
    conflict = replace(manifest, stream_sha256=os.urandom(32)).packet()
    with pytest.raises(QRVideoError, match="Conflicting manifest"):
        recover_stream(packets + [conflict], output)
    assert not output.exists()


def test_untrusted_manifest_limits(tmp_path: Path) -> None:
    manifest, _ = packets_for(tmp_path, b"test")
    output = tmp_path / "failed"
    with pytest.raises(QRVideoError, match="exceeds recovery byte limit"):
        recover_stream([replace(manifest, stream_size=MAX_STREAM_SIZE).packet()], output)
    with pytest.raises(QRVideoError, match="observed 0/"):
        recover_stream([manifest.packet()], output)
    with pytest.raises(QRVideoError, match="64 GiB"):
        replace(manifest, stream_size=MAX_STREAM_SIZE + 1).packet()
    raw = bytearray(manifest.packet())
    # Keep the outer CRC valid while corrupting the manifest's format version.
    raw[36 + 4] = 255
    raw[-4:] = struct.pack(">I", zlib.crc32(raw[:-4]))
    with pytest.raises(QRVideoError, match="manifest magic or version"):
        recover_stream([bytes(raw)], output)
    assert not output.exists()


def test_source_change_during_packet_encoding(tmp_path: Path) -> None:
    manifest, _ = packets_for(tmp_path, b"test")
    source = tmp_path / "source"
    source.write_bytes(b"more")
    with pytest.raises(QRVideoError, match="changed"):
        list(encode_packets(source, manifest))


def test_recovery_never_overwrites(tmp_path: Path) -> None:
    _, packets = packets_for(tmp_path, b"test")
    output = tmp_path / "existing"
    output.write_bytes(b"keep")
    with pytest.raises(QRVideoError, match="already exists"):
        recover_stream(packets, output)
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize(("k", "m", "interleave"), [(1, 1, 16), (4, 1, 2), (200, 55, 1)])
def test_custom_coding_parameters(
    tmp_path: Path,
    k: int,
    m: int,
    interleave: int,
) -> None:
    data = os.urandom(k * SHARD_SIZE + 19)
    _, packets = packets_for(tmp_path, data, k=k, m=m, interleave=interleave)
    decode_packets(tmp_path, list(reversed(packets)), data)
