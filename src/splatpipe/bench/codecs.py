"""Codecs measured by the rate-distortion benchmark."""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Protocol, runtime_checkable

from splatpipe.errors import ConfigError
from splatpipe.formats.splat import decode_splat, encode_splat
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply


def directory_size(directory: Path) -> int:
    """Return the recursive size of a codec directory."""
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


@runtime_checkable
class Codec(Protocol):
    """Encode into an owned directory and decode back to a GaussianCloud."""

    name: str

    def encode(self, cloud: GaussianCloud, directory: Path) -> None: ...

    def decode(self, directory: Path) -> GaussianCloud: ...

    def size(self, directory: Path) -> int: ...


class PlyCodec:
    """Lossless PLY anchor."""

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
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scene.splat").write_bytes(encode_splat(cloud, order=self.order))

    def decode(self, directory: Path) -> GaussianCloud:
        return decode_splat((Path(directory) / "scene.splat").read_bytes())

    def size(self, directory: Path) -> int:
        return directory_size(directory)


class PngCodec:
    """GPU-backed gsplat PngCompression baseline."""

    name = "png"

    def __init__(self, use_sort: bool = True, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {seed!r}")
        self.use_sort = use_sort
        self.seed = seed

    def _make_compressor(self):
        from gsplat.compression import PngCompression

        return PngCompression(use_sort=self.use_sort, verbose=False)

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        import torch
        from splatpipe.compress.sh import seeded_compression

        Path(directory).mkdir(parents=True, exist_ok=True)
        splats = {
            "means": torch.from_numpy(cloud.means).cuda(),
            "scales": torch.from_numpy(cloud.scales).cuda(),
            "quats": torch.from_numpy(cloud.quats).cuda(),
            "opacities": torch.from_numpy(cloud.opacities).cuda(),
            "sh0": torch.from_numpy(cloud.sh0).unsqueeze(1).cuda(),
            "shN": torch.from_numpy(cloud.shN).cuda(),
        }
        with (
            seeded_compression(self.seed),
            open(os.devnull, "w", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
        ):
            self._make_compressor().compress(str(directory), splats)

    def decode(self, directory: Path) -> GaussianCloud:
        splats = self._make_compressor().decompress(str(directory))
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


class ShVqCodec(PngCodec):
    """PngCompression with a smaller, seeded SH vector codebook."""

    def __init__(
        self,
        n_clusters: int,
        codebook_bits: int = 6,
        use_sort: bool = True,
        seed: int = 42,
    ) -> None:
        from splatpipe.compress.sh import label_dtype, validate_codebook_bits

        label_dtype(n_clusters)
        validate_codebook_bits(codebook_bits)
        self.n_clusters = n_clusters
        self.codebook_bits = codebook_bits
        super().__init__(use_sort=use_sort, seed=seed)
        self.name = f"shvq{n_clusters}"

    def _make_compressor(self):
        from splatpipe.compress.sh import make_sh_vq_compressor

        return make_sh_vq_compressor(
            n_clusters=self.n_clusters,
            codebook_bits=self.codebook_bits,
            use_sort=self.use_sort,
        )


class PrunedCodec:
    """Apply one fixed ranking before delegating to an existing codec."""

    def __init__(
        self,
        codec: Codec,
        scores,
        retained_percent: int,
        score_name: str,
    ) -> None:
        from splatpipe.compress.prune import parse_retained_percentages

        parse_retained_percentages(str(retained_percent))
        self.codec = codec
        self.scores = scores
        self.retained_percent = retained_percent
        self.name = f"{score_name}{retained_percent}-{codec.name}"

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        from splatpipe.compress.prune import prune_by_score

        pruned = prune_by_score(
            cloud,
            self.scores,
            retained_fraction=self.retained_percent / 100.0,
        )
        self.codec.encode(pruned, directory)

    def decode(self, directory: Path) -> GaussianCloud:
        return self.codec.decode(directory)

    def size(self, directory: Path) -> int:
        return self.codec.size(directory)


CODECS = {
    "ply": PlyCodec,
    "splat": SplatCodec,
    "png": PngCodec,
    "shvq256": lambda: ShVqCodec(256),
    "shvq1024": lambda: ShVqCodec(1024),
    "shvq4096": lambda: ShVqCodec(4096),
}


def build_codecs(names: str) -> list[Codec]:
    """Build codecs in measurement order; the first is the ratio anchor."""
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
