"""The staged container measurement: codec per block, then ordering, then layout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from splatpipe.errors import ArtifactError
from splatpipe.formats.container import (
    DEFAULT_CODEC,
    encode_block,
    pack_scene,
    write_container,
)
from splatpipe.gaussians import read_ply
from splatpipe.paths import RunPaths

CANDIDATE_CODECS = ("raw", "deflate", "png")


def measure_block_codecs(scene, candidates=CANDIDATE_CODECS) -> dict:
    """Stored size of every block under every candidate codec."""
    measured = {}
    for name, field in scene.fields.items():
        raw = np.ascontiguousarray(field.values).tobytes()
        measured[name] = {codec: len(encode_block(raw, codec)) for codec in candidates}
    return measured


def best_codecs(measurements: dict) -> dict:
    """Smallest per block. Ties go to raw, which costs the viewer no decode."""
    order = {"raw": 0, "deflate": 1, "png": 2}
    return {
        name: min(sizes, key=lambda codec: (sizes[codec], order.get(codec, 99)))
        for name, sizes in measurements.items()
    }


def _container_bytes(scene, codecs: dict, tmp: Path) -> int:
    write_container(
        scene, tmp, codecs={k: v for k, v in codecs.items() if k in scene.fields}
    )
    return tmp.stat().st_size


def measure_orders(cloud, orders, codecs, sh_vq=None, tmp_dir: Path | None = None) -> dict:
    measured = {}
    with tempfile.TemporaryDirectory(dir=tmp_dir) as scratch:
        target = Path(scratch) / "probe.splatc"
        for order in orders:
            scene = pack_scene(cloud, sh_vq=sh_vq, order=order)
            size = _container_bytes(scene, codecs, target)
            measured[order] = {
                "bytes": size,
                "count": scene.count,
                # Orders that crop store fewer Gaussians, so totals are not
                # comparable across this stage and this is what to compare.
                "bytes_per_gaussian": size / scene.count,
            }
    return measured


def measure_layouts(cloud, codecs, sh_vq=None, tmp_dir: Path | None = None) -> dict:
    """Per-attribute blocks against one interleaved block of the same values."""
    scene = pack_scene(cloud, sh_vq=sh_vq, order="morton")
    with tempfile.TemporaryDirectory(dir=tmp_dir) as scratch:
        target = Path(scratch) / "probe.splatc"
        struct_of_arrays = _container_bytes(scene, codecs, target)

    interleaved = np.concatenate(
        [
            np.ascontiguousarray(field.values)
            .reshape(scene.count, -1)
            .view(np.uint8)
            .reshape(scene.count, -1)
            for field in scene.fields.values()
            if len(field.values) == scene.count
        ],
        axis=1,
    )
    codec = codecs.get("means", DEFAULT_CODEC)
    return {
        "struct_of_arrays": struct_of_arrays,
        "array_of_structs": len(encode_block(interleaved.tobytes(), codec)),
    }


def measure_means_split(cloud, codecs, sh_vq=None, tmp_dir: Path | None = None) -> dict:
    """uint16 means interleaved, against separate high and low byte planes."""
    scene = pack_scene(cloud, sh_vq=sh_vq, order="morton")
    values = np.ascontiguousarray(scene.fields["means"].values)
    codec = codecs.get("means", DEFAULT_CODEC)
    interleaved = len(encode_block(values.tobytes(), codec))
    low = (values & 0xFF).astype(np.uint8)
    high = (values >> 8).astype(np.uint8)
    split = len(encode_block(low.tobytes(), codec)) + len(
        encode_block(high.tobytes(), codec)
    )
    return {"interleaved": interleaved, "split_hi_lo": split}


def run_pack_bench(
    run_dir: Path,
    orders: tuple[str, ...] = ("none", "morton"),
    candidates: tuple[str, ...] = CANDIDATE_CODECS,
    sh_codebook: Path | str | None = None,
) -> dict:
    paths = RunPaths(root=Path(run_dir))
    if not paths.ply.is_file():
        raise ArtifactError(f"{run_dir} holds no scene.ply at {paths.ply}")

    cloud = read_ply(paths.ply)
    paths.container_dir.mkdir(parents=True, exist_ok=True)

    sh_vq = None
    if sh_codebook is not None:
        from splatpipe.compress.sh import sh_vq_from_baseline

        sh_vq = sh_vq_from_baseline(sh_codebook, cloud.shN)

    scene = pack_scene(cloud, sh_vq=sh_vq, order="morton")
    block_codecs = measure_block_codecs(scene, candidates)
    selected = best_codecs(block_codecs)

    result = {
        "source_gaussians": len(cloud),
        "sh_codebook": None if sh_codebook is None else str(sh_codebook),
        "block_codecs": block_codecs,
        "best_codecs": selected,
        "orders": measure_orders(cloud, orders, selected, sh_vq),
        "layouts": measure_layouts(cloud, selected, sh_vq),
        "means_split": measure_means_split(cloud, selected, sh_vq),
        "selected_bytes": 0,
    }

    write_container(scene, paths.container, codecs=selected)
    result["selected_bytes"] = paths.container.stat().st_size
    paths.container_meta.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
