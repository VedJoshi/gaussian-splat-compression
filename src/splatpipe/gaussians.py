"""Struct-of-arrays container for a Gaussian splat scene, plus .ply I/O.

numpy only, deliberately no torch. This is what compress/ consumes, and keeping
it torch-free is what makes the compression tests run in milliseconds on CPU.

Units, which are the main trap here: `scales` are logarithms and `opacities`
are logits, exactly as stored in the .ply. Nothing is converted on read.
Consumers apply exp() and sigmoid() themselves, so there is one convention in
the codebase rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from splatpipe.errors import ArtifactError

SH_C0 = 0.28209479177387814

_ARRAY_SHAPES = {
    "means": ("N", 3),
    "scales": ("N", 3),
    "quats": ("N", 4),
    "opacities": ("N",),
    "sh0": ("N", 3),
    "shN": ("N", "K", 3),
}


@dataclass(frozen=True)
class GaussianCloud:
    means: np.ndarray  # (N, 3) float32, world space
    scales: np.ndarray  # (N, 3) float32, LOG scale
    quats: np.ndarray  # (N, 4) float32, (w, x, y, z), not normalised
    opacities: np.ndarray  # (N,) float32, LOGIT
    sh0: np.ndarray  # (N, 3) float32, the view-independent DC term
    shN: np.ndarray  # (N, K, 3) float32, K = 15 at degree 3

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        """Degree d has (d+1)^2 coefficients, one of which is the DC term."""
        return int(round((self.shN.shape[1] + 1) ** 0.5)) - 1

    def validate(self) -> None:
        n = len(self)
        for name, expected in _ARRAY_SHAPES.items():
            array = getattr(self, name)
            if array.dtype != np.float32:
                raise ArtifactError(f"{name} has dtype {array.dtype}, expected float32")
            if array.ndim != len(expected):
                raise ArtifactError(
                    f"{name} has shape {array.shape}, expected {len(expected)} dimensions"
                )
            if array.shape[0] != n:
                raise ArtifactError(
                    f"{name} has {array.shape[0]} rows but means has {n}"
                )
            for axis, size in enumerate(expected):
                if isinstance(size, int) and array.shape[axis] != size:
                    raise ArtifactError(
                        f"{name} has shape {array.shape}, expected axis {axis} to be {size}"
                    )
        if (self.shN.shape[1] + 1) ** 0.5 % 1 != 0:
            raise ArtifactError(
                f"shN has {self.shN.shape[1]} coefficients, which is not (d+1)^2 - 1 "
                f"for any integer degree d"
            )

    def take(self, index: np.ndarray) -> GaussianCloud:
        """Reorder or subset every array with the same index."""
        return replace(
            self,
            means=self.means[index],
            scales=self.scales[index],
            quats=self.quats[index],
            opacities=self.opacities[index],
            sh0=self.sh0[index],
            shN=self.shN[index],
        )


def _f_rest_names(k: int) -> list[str]:
    return [f"f_rest_{i}" for i in range(k * 3)]


def read_ply(path: Path | str) -> GaussianCloud:
    """Read a 3DGS .ply into a GaussianCloud.

    Reads by property name, never by byte offset: the field order in the file
    is x, y, z, f_dc, f_rest, opacity, scale, rot, which is not the order the
    struct is usually written down in.
    """
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names)

    def column(name: str) -> np.ndarray:
        if name not in names:
            raise ArtifactError(f"{path} has no property {name!r}")
        return np.asarray(vertex[name], dtype=np.float32)

    k = sum(1 for name in names if name.startswith("f_rest_")) // 3
    n = len(vertex)

    # Channel-major on disk: (N, 3, K) flattened. Transpose back to (N, K, 3).
    if k:
        flat = np.stack([column(name) for name in _f_rest_names(k)], axis=1)
        shN = flat.reshape(n, 3, k).transpose(0, 2, 1).copy()
    else:
        shN = np.zeros((n, 0, 3), dtype=np.float32)

    cloud = GaussianCloud(
        means=np.stack([column("x"), column("y"), column("z")], axis=1),
        scales=np.stack([column(f"scale_{i}") for i in range(3)], axis=1),
        quats=np.stack([column(f"rot_{i}") for i in range(4)], axis=1),
        opacities=column("opacity"),
        sh0=np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1),
        shN=shN,
    )
    cloud.validate()
    return cloud


def write_ply(cloud: GaussianCloud, path: Path | str) -> None:
    """Write a GaussianCloud as a 3DGS .ply, in the field order gsplat uses."""
    cloud.validate()
    n, k = len(cloud), cloud.shN.shape[1]

    columns: list[tuple[str, np.ndarray]] = [
        ("x", cloud.means[:, 0]),
        ("y", cloud.means[:, 1]),
        ("z", cloud.means[:, 2]),
        ("f_dc_0", cloud.sh0[:, 0]),
        ("f_dc_1", cloud.sh0[:, 1]),
        ("f_dc_2", cloud.sh0[:, 2]),
    ]
    if k:
        flat = cloud.shN.transpose(0, 2, 1).reshape(n, k * 3)
        columns += list(zip(_f_rest_names(k), flat.T))
    columns.append(("opacity", cloud.opacities))
    columns += [(f"scale_{i}", cloud.scales[:, i]) for i in range(3)]
    columns += [(f"rot_{i}", cloud.quats[:, i]) for i in range(4)]

    data = np.empty(n, dtype=[(name, "<f4") for name, _ in columns])
    for name, values in columns:
        data[name] = values

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
