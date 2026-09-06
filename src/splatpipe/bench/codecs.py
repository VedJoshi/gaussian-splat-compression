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

from splatpipe.errors import ConfigError
from splatpipe.formats.splat import decode_splat, encode_splat
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply


def directory_size(directory: Path) -> int:
    """Total bytes of every file under `directory`, recursively.

    Two kinds of path return 0 rather than raising, both because rglob yields
    nothing for them: a directory that does not exist, and a path that is a
    file. A caller that needs either to be an error has to check for itself.
    Every codec in this module creates its target directory in encode, so a 0
    from a codec's size() means the directory is there and empty.
    """
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


@runtime_checkable
class Codec(Protocol):
    """`name` is a scratch directory name as well as a label, so it has to be unique.

    `encode` owns its target directory and creates it when it is missing, rather
    than requiring the caller to. That is the half of the contract the
    implementations disagreed about: `write_ply` creates the parent itself, so
    PlyCodec worked into a missing directory while SplatCodec raised
    FileNotFoundError. Task 9's `run_bench` calls mkdir before every encode
    (plan line 1623), so that caller tolerates either convention. Task 6's
    test_render_uses_the_clouds_own_sh_degree does not: it encodes into an
    uncreated `tmp_path / "splat"` (plan line 848), which is the call the splat
    side raised on. That caller is what makes this mandatory rather than a
    preference.
    """

    name: str

    def encode(self, cloud: GaussianCloud, directory: Path) -> None: ...

    def decode(self, directory: Path) -> GaussianCloud: ...

    def size(self, directory: Path) -> int: ...


class PlyCodec:
    """The lossless anchor. Its point is the 1.00x origin of every ratio."""

    name = "ply"

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        # No mkdir here: write_ply creates the parent, which is what satisfies
        # the protocol's contract for this codec.
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
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scene.splat").write_bytes(encode_splat(cloud, order=self.order))

    def decode(self, directory: Path) -> GaussianCloud:
        return decode_splat((Path(directory) / "scene.splat").read_bytes())

    def size(self, directory: Path) -> int:
        return directory_size(directory)


class PngCodec:
    """gsplat's own PngCompression, the baseline this project has to beat.

    Quantises to PNG images with a PLAS spatial sort, and vector-quantises the
    higher-order harmonics with K-means. Needs cupy, torchpq and plas, and needs
    a GPU: there is no CPU path.
    """

    name = "png"

    def __init__(self, use_sort: bool = True) -> None:
        self.use_sort = use_sort

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        import torch
        from gsplat.compression import PngCompression

        Path(directory).mkdir(parents=True, exist_ok=True)
        # No .copy() here. from_numpy shares memory with the caller's array, but
        # .cuda() then allocates a separate device tensor, so nothing compress()
        # does can reach the caller. Measured against this gsplat build by
        # deleting the six .copy() calls the brief specified: all six arrays came
        # back byte-identical, so the copies were decorative.
        splats = {
            "means": torch.from_numpy(cloud.means).cuda(),
            "scales": torch.from_numpy(cloud.scales).cuda(),
            "quats": torch.from_numpy(cloud.quats).cuda(),
            "opacities": torch.from_numpy(cloud.opacities).cuda(),
            "sh0": torch.from_numpy(cloud.sh0).unsqueeze(1).cuda(),
            "shN": torch.from_numpy(cloud.shN).cuda(),
        }
        PngCompression(use_sort=self.use_sort, verbose=False).compress(
            str(directory), splats
        )

    def decode(self, directory: Path) -> GaussianCloud:
        from gsplat.compression import PngCompression

        splats = PngCompression(verbose=False).decompress(str(directory))
        cloud = GaussianCloud(
            means=splats["means"].cpu().numpy().astype("float32"),
            scales=splats["scales"].cpu().numpy().astype("float32"),
            quats=splats["quats"].cpu().numpy().astype("float32"),
            opacities=splats["opacities"].cpu().numpy().astype("float32"),
            sh0=splats["sh0"].squeeze(1).cpu().numpy().astype("float32"),
            shN=splats["shN"].cpu().numpy().astype("float32"),
        )
        cloud.validate()
        return cloud

    def size(self, directory: Path) -> int:
        return directory_size(directory)


CODECS = {
    "ply": PlyCodec,
    "splat": SplatCodec,
    "png": PngCodec,
}


def build_codecs(names: str) -> list:
    """Build codecs from a comma-separated name list, preserving order.

    Order is significant: the first codec is the anchor whose size is the
    denominator of every ratio, so it should be a lossless one.
    """
    selected = [name.strip() for name in names.split(",") if name.strip()]
    if not selected:
        raise ConfigError("no codecs selected")
    unknown = [name for name in selected if name not in CODECS]
    if unknown:
        raise ConfigError(
            f"unknown codec(s) {', '.join(unknown)}. "
            f"Known codecs are: {', '.join(sorted(CODECS))}"
        )
    return [CODECS[name]() for name in selected]
