from dataclasses import dataclass

from qr_video.errors import QRVideoError


@dataclass(frozen=True)
class Profile:
    name: str
    identifier: int
    version: int
    pixels_per_module: int
    qr_capacity: int
    width: int = 640
    height: int = 360
    fps: int = 30
    repeat: int = 3

    @property
    def side(self) -> int:
        return (17 + 4 * self.version + 8) * self.pixels_per_module

    @property
    def positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        gap = (self.width - 2 * self.side) // 3
        y = (self.height - self.side) // 2
        return ((gap, y), (self.width - gap - self.side, y))

    def validate(self) -> None:
        if (self.identifier, self.version, self.pixels_per_module, self.qr_capacity) != (
            1,
            20,
            3,
            382,
        ):
            raise QRVideoError("Unsupported QR geometry; use the conservative profile")
        if self.width != 640 or self.height != 360 or 2 * self.side > self.width:
            raise QRVideoError("QR codes and their four-module quiet zones must fit the frame")
        if not 1 <= self.fps <= 120 or not 1 <= self.repeat <= 120:
            raise QRVideoError("FPS and frame repeat must each be between 1 and 120")
        if self.fps % self.repeat:
            raise QRVideoError("FPS must be divisible by frame repeat")


CONSERVATIVE = Profile("conservative", 1, 20, 3, 382)
