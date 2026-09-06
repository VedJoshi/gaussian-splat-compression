"""The measured rate-distortion curve, and what makes a point traceable.

A point without its context is a number rather than a measurement, so every
curve carries the scene, the config digest, the held-out view count, the library
versions and the Git commit. Versions come from manifest.collect_versions rather
than a second copy, so a manifest and a curve written by the same run cannot
disagree about what produced them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from splatpipe.manifest import collect_versions


@dataclass(frozen=True)
class CurvePoint:
    codec: str
    bytes: int
    ratio: float
    gaussians: int
    psnr: float
    ssim: float
    lpips: float
    encode_seconds: float
    decode_seconds: float

    @classmethod
    def of(
        cls,
        codec: str,
        size_bytes: int,
        gaussians: int,
        scores: dict[str, float],
        encode_seconds: float,
        decode_seconds: float,
        anchor_bytes: int,
    ) -> CurvePoint:
        return cls(
            codec=codec,
            bytes=size_bytes,
            ratio=anchor_bytes / size_bytes,
            gaussians=gaussians,
            psnr=scores["psnr"],
            ssim=scores["ssim"],
            lpips=scores["lpips"],
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
        )


@dataclass
class Curve:
    scene: str
    config_digest: str
    held_out_views: int
    created_utc: str
    versions: dict[str, str] = field(default_factory=dict)
    points: list[CurvePoint] = field(default_factory=list)

    @classmethod
    def start(cls, scene: str, config_digest: str, held_out_views: int) -> Curve:
        return cls(
            scene=scene,
            config_digest=config_digest,
            held_out_views=held_out_views,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            versions=collect_versions(),
        )

    def add(self, point: CurvePoint) -> None:
        self.points.append(point)

    def write_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path | str) -> Curve:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["points"] = [CurvePoint(**point) for point in data["points"]]
        return cls(**data)

    def write_plot(self, path: Path | str) -> None:
        """Quality against size, size on a log axis because the range is 15x."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(figsize=(6, 4), dpi=150)
        for point in self.points:
            megabytes = point.bytes / 2**20
            axes.scatter(megabytes, point.psnr, s=40)
            axes.annotate(
                f"{point.codec} ({point.ratio:.2f}x)",
                (megabytes, point.psnr),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        axes.set_xscale("log")
        axes.set_xlabel("Size (MiB, log scale)")
        axes.set_ylabel("Held-out PSNR (dB)")
        axes.set_title(f"{self.scene}: rate against distortion")
        axes.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
