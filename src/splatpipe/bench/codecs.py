"""What a rate-distortion point is made of.

A codec turns a GaussianCloud into files on disk and back. The rate is the
total size of those files; the distortion is whatever rendering the decoded
cloud costs in image quality.

Directory-based rather than bytes-based because PngCompression genuinely
produces eight files, and flattening them into one blob would misreport the
rate, which is the single number this component exists to measure. A codec that
produces one buffer satisfies this protocol by writing one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from splatpipe.formats.splat import decode_splat, encode_splat
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply


def directory_size(directory: Path) -> int:
    """Total bytes of every file under `directory`, recursively."""
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


@runtime_checkable
class Codec(Protocol):
    name: str

    def encode(self, cloud: GaussianCloud, directory: Path) -> None: ...

    def decode(self, directory: Path) -> GaussianCloud: ...

    def size(self, directory: Path) -> int: ...


class PlyCodec:
    """The lossless anchor. Its point is the 1.00x origin of every ratio."""

    name = "ply"

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        write_ply(cloud, Path(directory) / "scene.ply")

    def decode(self, directory: Path) -> GaussianCloud:
        return read_ply(Path(directory) / "scene.ply")

    def size(self, directory: Path) -> int:
        return directory_size(directory)


class SplatCodec:
    """The 32-byte web format. Discards f_rest, so it decodes at sh_degree 0."""

    name = "splat"

    def __init__(self, order: str = "morton") -> None:
        self.order = order

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        path = Path(directory) / "scene.splat"
        path.write_bytes(encode_splat(cloud, order=self.order))

    def decode(self, directory: Path) -> GaussianCloud:
        return decode_splat((Path(directory) / "scene.splat").read_bytes())

    def size(self, directory: Path) -> int:
        return directory_size(directory)
