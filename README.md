# qr-video

A local Python CLI that carries arbitrary binary files in visible QR-code MP4
videos. The decoder reconstructs the file from a local video. Optional AES-256-GCM
encryption accepts a key value or a separate key file.

## Install

Use Python 3.11 or newer and FFmpeg with the `libx264` encoder. Both `ffmpeg` and
`ffprobe` must be on `PATH`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ffmpeg -version
qr-video --help
```

On macOS, `brew install ffmpeg` supplies both FFmpeg tools. On Debian or Ubuntu,
use the distribution's `ffmpeg` package. Do not install Python dependencies into
the operating system's Python.

## Encode and decode

Without encryption:

```sh
qr-video encode input.bin --out encoded.mp4
qr-video decode encoded.mp4 --out recovered.bin
```

With encryption:

```sh
qr-video keygen --out secret.key
KEY="$(cat secret.key)"
qr-video encode input.bin --out encoded.mp4 --key "$KEY"
qr-video decode encoded.mp4 --out recovered.bin --key "$KEY"
```

Compression is optional and happens before encryption. The envelope records the
choice, so decoding needs no compression flag:

```sh
qr-video encode input.bin --out encoded.mp4 --compress --key "$KEY"
qr-video decode encoded.mp4 --out recovered.bin --key "$KEY"
```

Get exact size and timing estimates without generating QR images:

```sh
qr-video stats input.bin
qr-video stats input.bin --compress --key "$KEY"
unset KEY
```

`stats` streams the input into a private temporary envelope to measure compression
and encryption overhead. It prints JSON to stdout. `encode` prints the same
statistics to stderr before generating video. `application_mb_per_hour` uses
the original file size and the actual scheduled video duration. MB means
1,000,000 bytes. Empty files report zero application MB/hour.

In an interactive terminal, `encode` shows the completed percentage and QR image
count. `decode` shows the number of video frames scanned. Redirected output and
non-interactive processes do not receive progress updates.

Every command returns a nonzero exit code on failure. Outputs appear only after
completion and verification. Existing outputs require `--overwrite`. Input and
key paths cannot be overwritten, including aliases through hard links. Key
generation never overwrites a file. Symlinks in parent directories work; output
leaves that are symlinks are rejected. The CLI fixes the resolved output parent
before processing, so retargeting a directory alias cannot redirect publication.
Default no-clobber publication requires hard-link support in the output
filesystem. Unsupported filesystems fail explicitly.

## Keep the key private

`keygen` uses operating-system randomness to generate exactly 32 ASCII letters
and digits with no trailing newline. On POSIX systems, the key file has mode
`0600`. This is the recommended way to create a key, but it is not a required
format. `--key` accepts any non-empty text and encodes it as UTF-8. `--key-file`
accepts any non-empty raw file. The options are mutually exclusive. The key is
never part of the video.

An inline key appears in the process arguments while the command runs. A literal
key may also remain in shell history. Expanding a shell variable avoids storing
the value in the command text, but it does not remove it from the process
arguments. Prefer `--key-file` on shared systems. Do not paste a real key into
chat, a README, or an issue.

Losing the key prevents decryption. Back it up separately from the video. Without encryption, anyone
with a readable video can recover the file. Encryption does not conceal the
presence of a transfer, its encoded size, or the QR transport parameters.
Compression can reveal information through the resulting size.

The decoder rejects a plaintext envelope when you pass `--key` or `--key-file`.
It does not ignore the key and accept an unauthenticated replacement.

Temporary directories are private. The decoder may write transient plaintext to
a restrictive temporary file while authenticating the stream. It publishes the
output only after all authentication, length, termination, and SHA-256 checks
pass. Wrong keys and damaged streams leave no new output and preserve any
existing destination. Cleanup unlinks temporary files; it does not promise
secure erasure from SSDs, swap, snapshots, or a compromised host.

## Physical profile and capacity

The shipped `conservative` profile uses two independent QR codes per image:

| Setting | Default |
| --- | --- |
| Video | 640x360, SDR grayscale content in H.264 `yuv420p` |
| QR version and correction | Version 20, level H |
| QR module scale | 3x3 pixels, integer rendering without antialiasing |
| Quiet zone | Four white modules around each code |
| Complete code size | 315x315 pixels |
| Code origins | Left x=3, right x=322, both y=22 |
| Frame rate | 30 fps |
| Image repeat | Three frames, giving 10 unique images/second |
| Encoder | `libx264`, CRF 18, preset `medium` |
| Outer coding | 100 data shards plus 25 parity shards |
| Interleave | Four coding blocks |

Version 20 H holds **382 raw byte-mode bytes per QR code**. A packet consumes
36 header bytes and four CRC bytes, leaving **342 envelope bytes per data
shard**. The payload stays binary throughout; neither Base64 nor a Unicode
conversion is involved.

A full coding block carries 34,200 envelope bytes in 125 QR packets. The encoder
adds two manifest packets at the start, two after every 20 data/parity packets,
and two at the end. It pads the final image with a blank second position when
the packet count is odd.

For a long transfer, the transport ceiling at default settings is:

```text
342 bytes * 2 codes * 10 images/second * (100/125) * (20/22)
    = 4,974.55 envelope bytes/second
    = 17.9084 envelope MB/hour
```

The file envelope, authenticated records, final partial block, initial/final
manifests, and incomplete final image reduce the original-file rate.
Compression may increase the rate measured in original bytes. Use `stats` for
the actual file rather than treating 382 bytes as application capacity.

Exact default schedules for uncompressed encrypted inputs:

| Original bytes | Envelope bytes | Data + parity shards | Video frames | Duration |
| --- | --- | --- | --- | --- |
| 0 | 173 | 1 + 1 | 9 | 0.3 s |
| 512 | 714 | 3 + 1 | 12 | 0.4 s |
| 4,097 | 4,299 | 13 + 4 | 33 | 1.1 s |
| 68,537 | 68,739 | 201 + 51 | 420 | 14.0 s |

The CLI exposes `--fps`, `--repeat`, `--data-shards`, `--parity-shards`, and
`--interleave`. FPS must be divisible by the repeat count. Coding requires
`1 <= k <= 200`, `1 <= m <= k`, and `k + m <= 255`. Interleave accepts 1 to 16.
Changes to timing, CRF, and redundancy can make recovery worse.

There is no dense profile in this release. Version 34 at two pixels/module
needs shared quiet-zone geometry to fit two codes in 640 pixels. This release
does not squeeze or clip codes to claim a higher capacity.

## Transport format

The implementation lives in `src/qr_video/transport.py`. Integers use network
byte order. Each QR contains one packet:

| Field | Bytes |
| --- | --- |
| Magic `QVP1` | 4 |
| Format version, currently 1 | 1 |
| Kind, data 0 or manifest 1 | 1 |
| Random transfer identifier | 16 |
| Coding block index | 4 |
| Shard index | 2 |
| Block data shard count | 2 |
| Block parity shard count | 2 |
| Shard size | 2 |
| Payload length | 2 |
| Payload | Up to 342 |
| CRC-32 of header and payload | 4 |

The 54-byte `QVM1` manifest payload records its version, profile, full-block
coding parameters, shard size, interleave depth, complete envelope length, and
SHA-256 of the envelope. Manifest packets have zero block/shard/coding fields.
Each manifest packet is 94 bytes including packet framing.

Manifest copies travel throughout the video, not just in its first frame.
The decoder checks their CRC and consistency. It can start without the leading
manifest and discover a later copy. Conflicting valid manifests or transfer
identifiers cause failure rather than choosing one.

The outer code uses `zfec`'s systematic Reed-Solomon erasure implementation.
Any 100 distinct valid shards from a full 125-shard block can reconstruct it.
The decoder discards failed packet CRCs as erasures. Repeated observations do
not add recovery capacity. Conflicting valid copies permanently erase that
shard identity, even if a later copy matches the first.

The last block uses `ceil(remaining_bytes/342)` data shards. Its parity count is
`ceil(last_k * full_m / full_k)`. The encoder zero-pads only the last data shard;
the manifest's exact stream length removes padding after recovery. Shortening
reduces tiny-file overhead without claiming 25 missing-shard tolerance for a
block that contains fewer than 125 shards.

Within each bounded group of blocks, the encoder emits shard 0 from each block,
then shard 1 from each, and continues until the group ends. The decoder stores
observations in a private SQLite database. It accepts arbitrary packet order
without keeping the full transfer in RAM, then reconstructs blocks in file
order. The final envelope SHA-256 detects a damaged reconstruction. CRC and
plain hashes are integrity checks, not authentication against an active attacker.

## File envelope and encryption

The version-1 envelope uses the `QRVENV01` magic. Its 56-byte header contains an
eight-byte magic, a one-byte version, one-byte flags, two reserved zero bytes,
a four-byte chunk size, an eight-byte original size, and a fresh 32-byte random
salt. Bit 0 selects encryption. Bit 1 selects zlib compression. The format does
not store the source filename or path.

Each record has a 13-byte prefix with a one-byte type, eight-byte sequential
index, and four-byte wire length. Data records contain up to 1,048,560 bytes
before encryption. Only the last data record can be shorter. AES-GCM appends
a 16-byte authentication tag, keeping a cipher record at or below 1 MiB.
Neither file size nor video length determines an in-memory allocation.

HKDF-SHA256 derives a separate AES-256 key for each file from the user key and
the random salt. Each record authenticates the complete header and its own
type, index, and length as associated data. Record indexes enforce ordering.
The final record has its own authentication tag.

The first 12 salt bytes provide a fresh 96-bit nonce base. The nonce for record
`i` is that base XOR `i`, encoded as 12 big-endian bytes. Data indexes start at
zero; the final index is the data-record count. This gives distinct nonces
within a file and fresh nonce bases when the same user key encrypts another
file. The per-file HKDF key also changes.

The final record's 88-byte plaintext contains the original size, compressed
payload size, data-record count, original SHA-256, and SHA-256 of the header and
preceding wire records. The decoder requires this record and rejects trailing
bytes. Deleting, reordering, or truncating records cannot publish partial
plaintext. Without encryption, these same hashes detect accidental corruption
but do not authenticate the sender.

For an uncompressed file of `N` bytes, let
`c = ceil(N / 1048560)`. The complete envelope is exactly
`N + 157 + 13*c` bytes without encryption, or `N + 173 + 29*c` bytes with
encryption. An empty encrypted file therefore has a 173-byte envelope. These
bytes all consume transport capacity.

## Resource limits and limitations

Encoding and decoding stream file data and video frames. They do not hold the
complete input, output video, or packet set in memory. Temporary disk space is
still required. Encoding stores an envelope and an MP4 before atomically copying
the video to its destination. Decoding stores packet observations, a recovered
envelope, and temporary plaintext. SQLite has additional per-packet disk overhead.
Set `TMPDIR` to a private location with enough space if needed.

The format limits both the original file and the complete envelope to 64 GiB.
Decoding defaults to a 1 GiB ceiling for each. Raise it only for a trusted
transfer when needed. Framing counts against the envelope limit, so an
uncompressed source at exactly 64 GiB cannot fit:

```sh
qr-video decode encoded.mp4 --out recovered.bin --key "$KEY" \
  --max-bytes 4294967296
```

At roughly 18 MB/hour before all file overhead, this is a slow way to carry
large files. QR level H corrects damage inside a detected code. The outer
Reed-Solomon code separately repairs missing packets. Neither can recover a
block with too few independent valid shards. More repeated video frames do not
increase the number of independent shards.

The decoder reads both grid positions independently and uses a QR detector.
It does not average unrelated frames or infer packet identity from frame
numbers. Keep the original 640x360 grid for the tested behavior. Cropping,
rescaling, overlays, blur, color conversion, and aggressive transcoding may
destroy codes. An unreadable or malformed video container can fail before
packet recovery.

Local video input allows MOV/MP4, Matroska/WebM, AVI, and MPEG-TS containers.
FFmpeg and ffprobe allow only local file/pipe protocols. Playlists and network
protocols are rejected. Frame dimensions cannot exceed 8,192 on either axis
or 8,294,400 total pixels.

## Local validation

On September 14, 2026, all 336 tests passed on macOS with Python 3.13.14 and
FFmpeg 8.1.1. Ruff and mypy also passed. The installed Python dependencies
included cryptography 48.0.1, NumPy 2.5.3, zfec 1.6.0.0, and zxing-cpp 3.1.1.

The CLI roundtrips cover empty files, all byte values including NUL and non-UTF8
bytes, compressible data, random incompressible data, and three-block files.
Each case runs with and without encryption. Tests inspect the real MP4's pixel
format, dimensions, frame rate, and frame count, then compare recovered bytes.

An encrypted 37,620-byte fixture also survived a second `libx264` encode at
CRF 23, preset `medium`, and `yuv420p`. That transcode removed the first nine
frames and changed 30 fps to 60 fps by duplicating frames. The removed frames
included both initial manifest copies. This command reproduces the complete
test, including the expected failure when a smaller last block loses more
shards than its parity budget permits:

```sh
pytest -q tests/test_cli.py::test_encrypted_transcode_with_dropped_and_duplicated_frames
```

The test applies this FFmpeg filter:

```text
select='gte(n,9)',setpts=N/(30*TB),fps=60
```

Packet tests separately remove exactly 25 of 125 full-block shards and recover
from the remaining 100. Removing 26 leaves only 99 and fails without output.
These tests call the outer erasure decoder directly, so QR-H correction cannot
hide a broken missing-packet recovery implementation.

A same-size synthetic benchmark encoded 1,347,325 random bytes in 15.3 seconds
on this machine. The 8,136-frame MP4 lasts 271.2 seconds and occupies 194.7 MB.
The native ZXing writer rendered each two-code image in 2.17 ms. The previous
Python writer took 58.8 ms, so QR rendering was about 27 times faster in this
benchmark. Payload density measures data per video duration, not storage
efficiency of the MP4 itself.

The QR tests also cover 382 NUL bytes through both lossy encodes. The encoder
uses ZXing's native QR writer with ECI disabled, which preserves the full raw
byte-mode capacity. Tests cover all-zero, all-byte-value, and random packets at
the 382-byte limit.

## Development

```sh
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
ruff check .
mypy src/qr_video
python -m build
twine check dist/*
```

Tests generate random keys and non-sensitive fixtures in temporary directories.
Real QR/FFmpeg tests have the `video` marker. They require FFmpeg and ffprobe;
`pytest -m "not video"` runs the remaining tests but does not verify video
transport. Generated keys, videos, build output, and `.venv` are ignored by Git.
