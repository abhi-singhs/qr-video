import hashlib
import os
import sqlite3
import struct
import tempfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import zfec

from qr_video.errors import QRVideoError
from qr_video.profiles import CONSERVATIVE, Profile

PACKET_MAGIC = b"QVP1"
VERSION = 1
DATA = 0
MANIFEST = 1
HEADER = struct.Struct(">4sBB16sIHHHHH")
CRC = struct.Struct(">I")
PACKET_OVERHEAD = HEADER.size + CRC.size
MANIFEST_STRUCT = struct.Struct(">4sBBHHHHQ32s")
MANIFEST_INTERVAL = 20
MAX_STREAM_SIZE = 64 * 1024**3
DEFAULT_MAX_BYTES = 1024**3
SHARD_SIZE = CONSERVATIVE.qr_capacity - PACKET_OVERHEAD


class InvalidPacket(QRVideoError):
    """An observation that must be discarded as an erasure."""


def validate_coding(k: int, m: int, interleave: int) -> None:
    if not 1 <= k <= 200 or not 1 <= m <= k or k + m > 255:
        raise QRVideoError("Coding requires 1 <= k <= 200, 1 <= m <= k, and k + m <= 255")
    if not 1 <= interleave <= 16:
        raise QRVideoError("Interleave depth must be between 1 and 16 blocks")


@dataclass(frozen=True)
class Packet:
    kind: int
    transfer_id: bytes
    block: int
    index: int
    k: int
    m: int
    shard_size: int
    payload: bytes

    def validate(self) -> None:
        if len(self.transfer_id) != 16 or not 0 <= self.block < 2**32:
            raise InvalidPacket("Invalid packet identity")
        if self.kind == MANIFEST:
            if (self.block, self.index, self.k, self.m, self.shard_size) != (0, 0, 0, 0, 0):
                raise InvalidPacket("Invalid manifest packet fields")
            if len(self.payload) != MANIFEST_STRUCT.size:
                raise InvalidPacket("Invalid manifest payload size")
        elif self.kind == DATA:
            if not 1 <= self.k <= 200 or not 1 <= self.m <= self.k or self.k + self.m > 255:
                raise InvalidPacket("Invalid shard coding parameters")
            if not 0 <= self.index < self.k + self.m:
                raise InvalidPacket("Shard index is outside its coding block")
            if self.shard_size != SHARD_SIZE or len(self.payload) != self.shard_size:
                raise InvalidPacket("Invalid shard payload size")
        else:
            raise InvalidPacket("Unsupported packet kind")

    def to_bytes(self) -> bytes:
        self.validate()
        body = (
            HEADER.pack(
                PACKET_MAGIC,
                VERSION,
                self.kind,
                self.transfer_id,
                self.block,
                self.index,
                self.k,
                self.m,
                self.shard_size,
                len(self.payload),
            )
            + self.payload
        )
        return body + CRC.pack(zlib.crc32(body))

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Packet":
        if not PACKET_OVERHEAD <= len(raw) <= CONSERVATIVE.qr_capacity:
            raise InvalidPacket("Invalid QR packet size")
        body = raw[: -CRC.size]
        if zlib.crc32(body) != CRC.unpack(raw[-CRC.size :])[0]:
            raise InvalidPacket("Packet CRC failed")
        magic, version, kind, identity, block, index, k, m, size, length = HEADER.unpack(
            body[: HEADER.size]
        )
        if magic != PACKET_MAGIC or version != VERSION:
            raise InvalidPacket("Unsupported QR packet magic or version")
        if length != len(body) - HEADER.size:
            raise InvalidPacket("Packet length does not match its header")
        packet = cls(kind, identity, block, index, k, m, size, body[HEADER.size :])
        packet.validate()
        return packet


@dataclass(frozen=True)
class Manifest:
    transfer_id: bytes
    stream_size: int
    stream_sha256: bytes
    k: int = 100
    m: int = 25
    interleave: int = 4
    profile_id: int = 1
    shard_size: int = SHARD_SIZE

    def validate(self) -> None:
        validate_coding(self.k, self.m, self.interleave)
        if len(self.transfer_id) != 16 or len(self.stream_sha256) != 32:
            raise QRVideoError("Invalid manifest identity or SHA-256")
        if self.profile_id != CONSERVATIVE.identifier or self.shard_size != SHARD_SIZE:
            raise QRVideoError("Unsupported manifest profile or shard size")
        if not 1 <= self.stream_size <= MAX_STREAM_SIZE:
            raise QRVideoError("Manifest stream size must be between 1 byte and 64 GiB")

    @property
    def blocks(self) -> int:
        block_size = self.k * self.shard_size
        return (self.stream_size + block_size - 1) // block_size

    def block_parameters(self, block: int) -> tuple[int, int, int]:
        if not 0 <= block < self.blocks:
            raise QRVideoError("Coding block is outside the manifest")
        size = min(self.k * self.shard_size, self.stream_size - block * self.k * self.shard_size)
        k = (size + self.shard_size - 1) // self.shard_size
        m = (k * self.m + self.k - 1) // self.k
        return k, m, size

    def packet(self) -> bytes:
        self.validate()
        payload = MANIFEST_STRUCT.pack(
            b"QVM1",
            VERSION,
            self.profile_id,
            self.k,
            self.m,
            self.shard_size,
            self.interleave,
            self.stream_size,
            self.stream_sha256,
        )
        return Packet(MANIFEST, self.transfer_id, 0, 0, 0, 0, 0, payload).to_bytes()

    @classmethod
    def from_packet(cls, packet: Packet) -> "Manifest":
        packet.validate()
        if packet.kind != MANIFEST:
            raise QRVideoError("Expected a manifest packet")
        magic, version, profile, k, m, size, interleave, length, digest = MANIFEST_STRUCT.unpack(
            packet.payload
        )
        if magic != b"QVM1" or version != VERSION:
            raise QRVideoError("Unsupported manifest magic or version")
        manifest = cls(packet.transfer_id, length, digest, k, m, interleave, profile, size)
        manifest.validate()
        return manifest


def prepare_manifest(
    source: Path,
    *,
    k: int = 100,
    m: int = 25,
    interleave: int = 4,
) -> Manifest:
    validate_coding(k, m, interleave)
    with source.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if not 1 <= size <= MAX_STREAM_SIZE:
            raise QRVideoError("Transport stream size must be between 1 byte and 64 GiB")
        digest = hashlib.file_digest(stream, "sha256").digest()
    manifest = Manifest(os.urandom(16), size, digest, k, m, interleave)
    manifest.validate()
    return manifest


def _data_packets(source: Path, manifest: Manifest) -> Iterator[bytes]:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for first in range(0, manifest.blocks, manifest.interleave):
            group: list[tuple[int, int, int, list[bytes]]] = []
            for block in range(first, min(first + manifest.interleave, manifest.blocks)):
                k, m, size = manifest.block_parameters(block)
                data = stream.read(size)
                if len(data) != size:
                    raise QRVideoError("Transport source was truncated during encoding")
                digest.update(data)
                data = data.ljust(k * manifest.shard_size, b"\x00")
                pieces = [
                    data[offset : offset + manifest.shard_size]
                    for offset in range(0, len(data), manifest.shard_size)
                ]
                shards = zfec.Encoder(k, k + m).encode(pieces)
                group.append((block, k, m, shards))
            for index in range(max(k + m for _, k, m, _ in group)):
                for block, k, m, shards in group:
                    if index < k + m:
                        yield Packet(
                            DATA,
                            manifest.transfer_id,
                            block,
                            index,
                            k,
                            m,
                            manifest.shard_size,
                            shards[index],
                        ).to_bytes()
        if stream.read(1) or digest.digest() != manifest.stream_sha256:
            raise QRVideoError("Transport source changed during encoding")


def encode_packets(source: Path, manifest: Manifest) -> Iterator[bytes]:
    manifest.validate()
    bootstrap = manifest.packet()
    yield bootstrap
    yield bootstrap
    for count, packet in enumerate(_data_packets(source, manifest), 1):
        yield packet
        if count % MANIFEST_INTERVAL == 0:
            yield bootstrap
            yield bootstrap
    yield bootstrap
    yield bootstrap


def transport_stats(manifest: Manifest, profile: Profile = CONSERVATIVE) -> dict[str, int | float]:
    manifest.validate()
    profile.validate()
    full = manifest.blocks - 1
    last_k, last_m, _ = manifest.block_parameters(full)
    data_shards = full * manifest.k + last_k
    parity_shards = full * manifest.m + last_m
    coded_packets = data_shards + parity_shards
    manifest_packets = 4 + 2 * (coded_packets // MANIFEST_INTERVAL)
    images = (coded_packets + manifest_packets + 1) // 2
    seconds = images * profile.repeat / profile.fps
    return {
        "qr_capacity_bytes": profile.qr_capacity,
        "packet_overhead_bytes": PACKET_OVERHEAD,
        "shard_payload_bytes": manifest.shard_size,
        "stream_bytes": manifest.stream_size,
        "blocks": manifest.blocks,
        "data_shards": data_shards,
        "parity_shards": parity_shards,
        "outer_redundancy_percent": 100 * parity_shards / data_shards,
        "manifest_packets": manifest_packets,
        "qr_packets": coded_packets + manifest_packets,
        "images": images,
        "video_frames": images * profile.repeat,
        "duration_seconds": seconds,
        "stream_mb_per_hour": manifest.stream_size / seconds * 3600 / 1_000_000,
    }


@dataclass
class RecoveryStats:
    observations: int = 0
    invalid_packets: int = 0
    duplicates: int = 0
    conflicts: int = 0
    unique_shards: int = 0
    blocks: int = 0


def recover_stream(
    observations: Iterable[bytes],
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[Manifest, RecoveryStats]:
    """Collect on disk so arbitrary reordering and late conflicts stay safe."""
    if not 1 <= max_bytes <= MAX_STREAM_SIZE:
        raise QRVideoError("Recovery byte limit must be between 1 byte and 64 GiB")
    if destination.exists() or destination.is_symlink():
        raise QRVideoError(f"Recovery destination already exists: {destination}")
    stats = RecoveryStats()
    with tempfile.TemporaryDirectory(prefix="qr-video-packets-") as directory:
        database = Path(directory) / "packets.sqlite"
        database.touch(mode=0o600)
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute(
                "CREATE TABLE shards (block INTEGER, idx INTEGER, k INTEGER, m INTEGER, "
                "data BLOB, conflict INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (block,idx))"
            )
            manifest = _collect(observations, connection, max_bytes, stats)
            _recover(connection, manifest, destination, stats)
        finally:
            connection.close()
    return manifest, stats


def _collect(
    observations: Iterable[bytes],
    connection: sqlite3.Connection,
    max_bytes: int,
    stats: RecoveryStats,
) -> Manifest:
    manifest: Manifest | None = None
    identity: bytes | None = None
    max_shards = 2 * ((max_bytes + SHARD_SIZE - 1) // SHARD_SIZE) + 510
    max_blocks = (max_bytes + SHARD_SIZE - 1) // SHARD_SIZE
    for raw in observations:
        stats.observations += 1
        try:
            packet = Packet.from_bytes(raw)
        except InvalidPacket:
            stats.invalid_packets += 1
            continue
        if identity is not None and identity != packet.transfer_id:
            raise QRVideoError("Video contains conflicting transfer identities")
        identity = packet.transfer_id
        if packet.kind == MANIFEST:
            candidate = Manifest.from_packet(packet)
            if candidate.stream_size > max_bytes:
                raise QRVideoError("Manifest exceeds recovery byte limit; adjust --max-bytes")
            if manifest is not None and manifest != candidate:
                raise QRVideoError("Conflicting manifest copies")
            manifest = candidate
            continue
        if packet.block >= max_blocks:
            raise QRVideoError("Shard block exceeds recovery byte limit")
        existing = connection.execute(
            "SELECT k,m,data,conflict FROM shards WHERE block=? AND idx=?",
            (packet.block, packet.index),
        ).fetchone()
        if existing is not None:
            stats.duplicates += 1
            if not existing[3] and existing[:3] != (packet.k, packet.m, packet.payload):
                connection.execute(
                    "UPDATE shards SET data=NULL,conflict=1 WHERE block=? AND idx=?",
                    (packet.block, packet.index),
                )
                stats.conflicts += 1
            continue
        if stats.unique_shards >= max_shards:
            raise QRVideoError("Too many distinct shards for recovery byte limit")
        connection.execute(
            "INSERT INTO shards (block,idx,k,m,data) VALUES (?,?,?,?,?)",
            (packet.block, packet.index, packet.k, packet.m, packet.payload),
        )
        stats.unique_shards += 1
        if stats.unique_shards % 1000 == 0:
            connection.commit()
    connection.commit()
    if manifest is None:
        raise QRVideoError("No valid manifest recovered from video")
    return manifest


def _recover(
    connection: sqlite3.Connection,
    manifest: Manifest,
    destination: Path,
    stats: RecoveryStats,
) -> None:
    invalid_blocks = connection.execute(
        "SELECT COUNT(*) FROM shards WHERE block >= ?",
        (manifest.blocks,),
    ).fetchone()[0]
    if invalid_blocks:
        raise QRVideoError("Shard block count conflicts with manifest")
    observed_blocks = connection.execute("SELECT COUNT(DISTINCT block) FROM shards").fetchone()[0]
    if observed_blocks != manifest.blocks:
        raise QRVideoError(
            f"Insufficient recovery packets: observed {observed_blocks}/{manifest.blocks} blocks"
        )
    digest = hashlib.sha256()
    # This is a private intermediate envelope, never the final recovered user file.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            for block in range(manifest.blocks):
                k, m, size = manifest.block_parameters(block)
                mismatches = connection.execute(
                    "SELECT COUNT(*) FROM shards WHERE block=? AND conflict=0 "
                    "AND (k!=? OR m!=? OR idx>=?)",
                    (block, k, m, k + m),
                ).fetchone()[0]
                if mismatches:
                    raise QRVideoError("Shard coding parameters conflict with manifest")
                rows = connection.execute(
                    "SELECT idx,data FROM shards WHERE block=? AND conflict=0 ORDER BY idx LIMIT ?",
                    (block, k),
                ).fetchall()
                if len(rows) < k:
                    raise QRVideoError(
                        f"Insufficient recovery packets for block {block}: "
                        f"need {k}, got {len(rows)}"
                    )
                decoded = zfec.Decoder(k, k + m).decode(
                    [row[1] for row in rows],
                    [row[0] for row in rows],
                )
                data = b"".join(decoded)[:size]
                digest.update(data)
                output.write(data)
                stats.blocks += 1
            if digest.digest() != manifest.stream_sha256:
                raise QRVideoError("Recovered stream SHA-256 failed")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
