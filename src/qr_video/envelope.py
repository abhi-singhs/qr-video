"""Version-1 streaming envelope with optional zlib and AES-256-GCM.

All integers use network byte order. The 56-byte header contains magic, version,
flags, a zero reserved field, chunk size, original size, and a random 32-byte salt.
Each record has a 13-byte type/index/wire-length prefix. Data records contain at
most 1 MiB minus the 16-byte GCM tag. Only the last data record may be shorter.
The final record contains original size, payload size, data record count, original
SHA-256, and a SHA-256 of the header and preceding wire records.

HKDF-SHA256 derives each file's AES key from the user key and salt. Every record
authenticates the full header and its prefix. The nonce is the first 12 salt bytes
XOR the record index, both interpreted as unsigned 96-bit big-endian integers.
Data indices start at zero; the final-record index is the data record count.
This mapping is injective within each file and uses a fresh random nonce base
for each encryption. Supplying a key for plaintext is rejected. Plaintext
checksums detect corruption but do not prevent deliberate forgery.

The original file limit is 64 GiB. Decoding defaults to a 1 GiB original-size
ceiling; max_bytes accepts integers from 1 through 64 GiB. The decoder checks the
header before reading records or creating output. Compressed payloads cannot
exceed zlib's bound for the declared original size. Decompression uses 64 KiB
output buffers and never writes beyond the declared size or configured ceiling.
Stored envelopes can exceed the original-size limit because of framing and
compression overhead. No source names or paths enter the envelope.
"""

import hashlib
import os
import secrets
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from qr_video.errors import QRVideoError
from qr_video.safeio import KEY_SIZE, atomic_output, check_output, open_input

MAGIC = b"QRVENV01"
VERSION = 1
FLAG_ENCRYPTED = 1
FLAG_COMPRESSED = 2
TAG_SIZE = 16
SALT_SIZE = 32
NONCE_SIZE = 12
CHUNK_SIZE = 1024 * 1024 - TAG_SIZE
IO_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_BYTES = 1024**3
MAX_FILE_SIZE = 64 * 1024**3
HEADER_STRUCT = struct.Struct(">8sBBHIQ32s")
RECORD_STRUCT = struct.Struct(">BQI")
FINAL_STRUCT = struct.Struct(">QQQ32s32s")
DATA_RECORD = 0
FINAL_RECORD = 1
KDF_INFO = b"qr-video/envelope/v1/aes-256-gcm"
AAD_DOMAIN = b"qr-video/envelope/v1/record\0"


def _payload_bound(original_size: int, compressed: bool) -> int:
    if not compressed:
        return original_size
    return (
        original_size + (original_size >> 12) + (original_size >> 14) + (original_size >> 25) + 13
    )


MAX_PAYLOAD_SIZE = _payload_bound(MAX_FILE_SIZE, True)
MAX_DATA_RECORDS = (MAX_PAYLOAD_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
MAX_ENVELOPE_SIZE = (
    HEADER_STRUCT.size
    + MAX_PAYLOAD_SIZE
    + (MAX_DATA_RECORDS + 1) * (RECORD_STRUCT.size + TAG_SIZE)
    + FINAL_STRUCT.size
)


@dataclass(frozen=True)
class EnvelopeInfo:
    original_size: int
    stored_size: int
    encrypted: bool
    compressed: bool
    sha256: str


def _validate_key(key: bytes | None) -> None:
    if key is not None and (not isinstance(key, bytes) or len(key) != KEY_SIZE):
        raise QRVideoError(f"Encryption key must be exactly {KEY_SIZE} raw bytes.")


def _cipher(key: bytes | None, salt: bytes) -> AESGCM | None:
    if key is None:
        return None
    derived = HKDF(algorithm=hashes.SHA256(), length=KEY_SIZE, salt=salt, info=KDF_INFO).derive(key)
    return AESGCM(derived)


def _record_nonce(salt: bytes, index: int) -> bytes:
    return (int.from_bytes(salt[:NONCE_SIZE], "big") ^ index).to_bytes(NONCE_SIZE, "big")


def _record_bytes(
    record_type: int,
    index: int,
    data: bytes,
    header: bytes,
    salt: bytes,
    cipher: AESGCM | None,
) -> bytes:
    length = len(data) + (TAG_SIZE if cipher is not None else 0)
    prefix = RECORD_STRUCT.pack(record_type, index, length)
    if cipher is not None:
        data = cipher.encrypt(_record_nonce(salt, index), data, AAD_DOMAIN + header + prefix)
    return prefix + data


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise QRVideoError("Truncated envelope.")
    return data


def encode_envelope(
    source: Path, destination: Path, *, key: bytes | None = None, compress: bool = False
) -> EnvelopeInfo:
    """Stream a regular source into a new envelope. Never replace the destination."""
    _validate_key(key)
    check_output(destination, protected=(source,))
    try:
        with open_input(source) as incoming:
            declared_size = os.fstat(incoming.fileno()).st_size
            if not 0 <= declared_size <= MAX_FILE_SIZE:
                raise QRVideoError(f"Original file exceeds the {MAX_FILE_SIZE}-byte limit.")
            flags = (FLAG_ENCRYPTED if key is not None else 0) | (
                FLAG_COMPRESSED if compress else 0
            )
            salt = secrets.token_bytes(SALT_SIZE)
            header = HEADER_STRUCT.pack(MAGIC, VERSION, flags, 0, CHUNK_SIZE, declared_size, salt)
            cipher = _cipher(key, salt)
            original_hash = hashlib.sha256()
            transcript = hashlib.sha256(header)
            compressor = zlib.compressobj() if compress else None
            original_size = 0
            payload_size = 0
            chunk_count = 0
            pending = bytearray()
            with atomic_output(destination) as outgoing:
                outgoing.write(header)

                def write_data(data: bytes) -> None:
                    nonlocal chunk_count
                    record = _record_bytes(DATA_RECORD, chunk_count, data, header, salt, cipher)
                    outgoing.write(record)
                    transcript.update(record)
                    chunk_count += 1

                def feed(data: bytes) -> None:
                    nonlocal payload_size
                    payload_size += len(data)
                    if payload_size > _payload_bound(declared_size, compress):
                        raise QRVideoError("Payload exceeds its declared size bound.")
                    pending.extend(data)
                    while len(pending) >= CHUNK_SIZE:
                        write_data(bytes(pending[:CHUNK_SIZE]))
                        del pending[:CHUNK_SIZE]

                while data := incoming.read(IO_CHUNK_SIZE):
                    original_size += len(data)
                    if original_size > declared_size:
                        raise QRVideoError("Source size changed while encoding.")
                    original_hash.update(data)
                    feed(compressor.compress(data) if compressor is not None else data)
                if original_size != declared_size:
                    raise QRVideoError("Source size changed while encoding.")
                if compressor is not None:
                    feed(compressor.flush())
                if pending:
                    write_data(bytes(pending))
                final = FINAL_STRUCT.pack(
                    original_size,
                    payload_size,
                    chunk_count,
                    original_hash.digest(),
                    transcript.digest(),
                )
                outgoing.write(
                    _record_bytes(FINAL_RECORD, chunk_count, final, header, salt, cipher)
                )
                stored_size = outgoing.tell()
            return EnvelopeInfo(
                original_size, stored_size, key is not None, compress, original_hash.hexdigest()
            )
    except (OSError, zlib.error, struct.error) as exc:
        raise QRVideoError(f"Cannot encode envelope: {exc}") from exc


def decode_envelope(
    source: Path,
    destination: Path,
    *,
    key: bytes | None = None,
    overwrite: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> EnvelopeInfo:
    """Validate within max_bytes before publishing complete plaintext."""
    _validate_key(key)
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or not 1 <= max_bytes <= MAX_FILE_SIZE
    ):
        raise QRVideoError(f"max_bytes must be an integer from 1 to {MAX_FILE_SIZE}.")
    check_output(destination, overwrite=overwrite, protected=(source,))
    try:
        with open_input(source) as incoming:
            stored_size = os.fstat(incoming.fileno()).st_size
            if stored_size > MAX_ENVELOPE_SIZE:
                raise QRVideoError("Envelope exceeds the maximum stored size.")
            header = _read_exact(incoming, HEADER_STRUCT.size)
            magic, version, flags, reserved, chunk_size, declared_size, salt = HEADER_STRUCT.unpack(
                header
            )
            if (
                magic != MAGIC
                or version != VERSION
                or flags & ~(FLAG_ENCRYPTED | FLAG_COMPRESSED)
                or reserved != 0
                or chunk_size != CHUNK_SIZE
                or declared_size > MAX_FILE_SIZE
            ):
                raise QRVideoError("Invalid or unsupported envelope header.")
            if declared_size > max_bytes:
                raise QRVideoError(
                    f"Declared original size exceeds the {max_bytes}-byte output limit."
                )
            encrypted = bool(flags & FLAG_ENCRYPTED)
            compressed = bool(flags & FLAG_COMPRESSED)
            if encrypted and key is None:
                raise QRVideoError("This encrypted envelope requires a 32-byte key.")
            if not encrypted and key is not None:
                raise QRVideoError("A key was supplied for an unencrypted envelope.")
            cipher = _cipher(key, salt)
            decompressor = zlib.decompressobj() if compressed else None
            original_hash = hashlib.sha256()
            transcript = hashlib.sha256(header)
            original_size = 0
            payload_size = 0
            chunk_count = 0
            short_record = False
            payload_limit = _payload_bound(declared_size, compressed)
            plaintext_limit = min(declared_size, max_bytes)

            with atomic_output(destination, overwrite=overwrite) as outgoing:

                def write_plaintext(data: bytes) -> None:
                    nonlocal original_size
                    if len(data) > plaintext_limit - original_size:
                        raise QRVideoError(
                            "Plaintext exceeds the declared size or configured output limit."
                        )
                    outgoing.write(data)
                    original_hash.update(data)
                    original_size += len(data)

                while True:
                    prefix = _read_exact(incoming, RECORD_STRUCT.size)
                    record_type, index, length = RECORD_STRUCT.unpack(prefix)
                    if index != chunk_count:
                        raise QRVideoError("Envelope record order is invalid.")
                    plain_length = length - (TAG_SIZE if encrypted else 0)
                    if record_type == DATA_RECORD:
                        if (
                            short_record
                            or not 0 < plain_length <= CHUNK_SIZE
                            or payload_size + plain_length > payload_limit
                            or chunk_count >= MAX_DATA_RECORDS
                        ):
                            raise QRVideoError("Invalid envelope data record length.")
                    elif record_type != FINAL_RECORD or plain_length != FINAL_STRUCT.size:
                        raise QRVideoError("Invalid envelope final record.")
                    wire_data = _read_exact(incoming, length)
                    data = wire_data
                    if cipher is not None:
                        try:
                            data = cipher.decrypt(
                                _record_nonce(salt, index),
                                wire_data,
                                AAD_DOMAIN + header + prefix,
                            )
                        except InvalidTag as exc:
                            raise QRVideoError(
                                "Envelope authentication failed. The key or data is incorrect."
                            ) from exc
                    if record_type == FINAL_RECORD:
                        final_size, final_payload, final_count, final_hash, final_transcript = (
                            FINAL_STRUCT.unpack(data)
                        )
                        if (
                            final_size != declared_size
                            or final_size != original_size
                            or final_payload != payload_size
                            or final_count != chunk_count
                            or final_hash != original_hash.digest()
                            or final_transcript != transcript.digest()
                        ):
                            raise QRVideoError("Envelope final size, count, or checksum mismatch.")
                        if decompressor is not None and (
                            not decompressor.eof
                            or decompressor.unused_data
                            or decompressor.unconsumed_tail
                        ):
                            raise QRVideoError("Incomplete or trailing compressed stream.")
                        if incoming.read(1) or incoming.tell() != stored_size:
                            raise QRVideoError("Unexpected bytes after the envelope final record.")
                        break
                    transcript.update(prefix)
                    transcript.update(wire_data)
                    payload_size += len(data)
                    chunk_count += 1
                    short_record = len(data) < CHUNK_SIZE
                    if decompressor is None:
                        write_plaintext(data)
                        continue
                    if decompressor.eof:
                        raise QRVideoError("Unexpected data after the compressed stream.")
                    while True:
                        output_limit = min(IO_CHUNK_SIZE, plaintext_limit - original_size + 1)
                        plain = decompressor.decompress(data, output_limit)
                        if decompressor.unused_data:
                            raise QRVideoError("Unexpected data after the compressed stream.")
                        write_plaintext(plain)
                        data = decompressor.unconsumed_tail
                        if decompressor.eof or (not data and len(plain) < output_limit):
                            break
            return EnvelopeInfo(
                original_size, stored_size, encrypted, compressed, original_hash.hexdigest()
            )
    except (OSError, zlib.error, struct.error) as exc:
        raise QRVideoError(f"Cannot decode envelope: {exc}") from exc
