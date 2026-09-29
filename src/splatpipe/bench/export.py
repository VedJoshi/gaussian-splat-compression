"""Write the deployable pruned container and score exactly those bytes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splatpipe.bench.codecs import ContainerCodec, PlyCodec, PrunedCodec
from splatpipe.bench.pack import best_codecs, measure_block_codecs
from splatpipe.bench.run import run_bench
from splatpipe.compress import sh
from splatpipe.compress.prune import prune_by_score
from splatpipe.errors import ArtifactError
from splatpipe.formats.container import pack_scene, write_container
from splatpipe.gaussians import read_ply
from splatpipe.paths import RunPaths


def run_export(
    run_dir: Path,
    scene_dir: Path,
    retained_percent: int = 80,
    device: str = "cuda",
    data_factor: int = 1,
    test_every: int = 8,
) -> dict:
    paths = RunPaths(root=Path(run_dir))
    output = paths.root / "export"
    if output.exists():
        raise ArtifactError(f"{output} exists; move it aside to re-export")
    if not paths.prune_scores.is_file():
        raise ArtifactError(f"no pruning scores at {paths.prune_scores}; run splatpipe prune first")
    codebook = paths.pruning_dir / "bench" / f"prune{retained_percent}-shvq4096"
    if not codebook.is_dir():
        raise ArtifactError(f"no SH codebook at {codebook}; run splatpipe prune with --retain {retained_percent}")

    with np.load(paths.prune_scores, allow_pickle=False) as archive:
        scores = archive["score"]
    cloud = read_ply(paths.ply)
    pruned = prune_by_score(cloud, scores, retained_percent / 100.0)
    scene = pack_scene(pruned, sh_vq=sh.sh_vq_from_baseline(codebook, pruned.shN))
    codecs = best_codecs(measure_block_codecs(scene))

    output.mkdir()
    container = output / "scene.splatc"
    write_container(scene, container, codecs=codecs)
    curve = run_bench(
        paths.root,
        scene_dir,
        [PlyCodec(), PrunedCodec(ContainerCodec(codecs=codecs, sh_codebook=codebook), scores, retained_percent, "prune")],
        device=device,
        data_factor=data_factor,
        test_every=test_every,
        output_namespace="export",
    )
    scored = output / "bench" / f"prune{retained_percent}-container" / "scene.splatc"
    if scored.read_bytes() != container.read_bytes():
        raise ArtifactError(f"{scored} differs from {container}; the curve does not describe the deployed file")

    result = {
        "source_gaussians": len(cloud),
        "retained_percent": retained_percent,
        "retained_gaussians": len(pruned),
        "sh_codebook": str(codebook),
        "codecs": codecs,
        "bytes": container.stat().st_size,
        "held_out_views": curve.held_out_views,
    }
    (output / "meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
