from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe.bench import pack as pack_bench
from splatpipe.errors import ArtifactError
from splatpipe.paths import RunPaths
from tests.test_bench_cpu import a_run
from tests.test_codecs import a_cloud


def test_block_codec_measurement_covers_every_block_and_candidate():
    from splatpipe.formats.container import pack_scene

    scene = pack_scene(a_cloud(300), order="none")
    measured = pack_bench.measure_block_codecs(scene, ("raw", "deflate"))
    assert set(measured) == set(scene.fields)
    for name, sizes in measured.items():
        assert set(sizes) == {"raw", "deflate"}, name
        assert all(value > 0 for value in sizes.values()), name


def test_best_codecs_picks_the_smallest_per_block():
    measurements = {
        "means": {"raw": 100, "deflate": 90},
        "quats": {"raw": 50, "deflate": 70},
    }
    assert pack_bench.best_codecs(measurements) == {"means": "deflate", "quats": "raw"}


def test_best_codecs_breaks_ties_toward_the_cheaper_decode():
    # raw needs no decompression, so an exact tie should not cost the viewer work.
    assert pack_bench.best_codecs({"means": {"raw": 100, "deflate": 100}}) == {
        "means": "raw"
    }


def test_order_measurement_reports_bytes_per_gaussian():
    cloud = a_cloud(300)
    measured = pack_bench.measure_orders(cloud, ("none", "morton"), {"means": "deflate"})
    assert set(measured) == {"none", "morton"}
    for name, entry in measured.items():
        assert entry["count"] == 300, name
        assert entry["bytes"] > 0, name
        assert entry["bytes_per_gaussian"] == pytest.approx(
            entry["bytes"] / entry["count"]
        ), name


def test_layout_measurement_compares_both_arrangements():
    measured = pack_bench.measure_layouts(a_cloud(300), {})
    assert set(measured) == {"struct_of_arrays", "array_of_structs"}
    assert all(value > 0 for value in measured.values())


def test_means_split_measurement_compares_both_arrangements():
    measured = pack_bench.measure_means_split(a_cloud(300), {})
    assert set(measured) == {"interleaved", "split_hi_lo"}
    assert all(value > 0 for value in measured.values())


def test_pack_bench_writes_its_metadata_and_leaves_other_curves_alone(tmp_path):
    paths = a_run(tmp_path, n=300)
    (paths.root / "curve.json").write_text('{"sentinel": true}', encoding="utf-8")

    result = pack_bench.run_pack_bench(
        paths.root, orders=("none", "morton"), candidates=("raw", "deflate")
    )

    assert set(result) == {
        "block_codecs",
        "best_codecs",
        "orders",
        "layouts",
        "means_split",
        "selected_bytes",
        "sh_codebook",
        "source_gaussians",
    }
    meta = json.loads(paths.container_meta.read_text(encoding="utf-8"))
    assert meta["best_codecs"] == result["best_codecs"]
    assert paths.container.is_file()
    assert json.loads((paths.root / "curve.json").read_text(encoding="utf-8")) == {
        "sentinel": True
    }


def test_pack_bench_requires_a_trained_ply(tmp_path):
    with pytest.raises(ArtifactError, match="scene.ply"):
        pack_bench.run_pack_bench(
            tmp_path / "empty", orders=("none",), candidates=("raw",)
        )


def test_bench_namespace_flag_keeps_the_existing_curve(tmp_path, monkeypatch):
    """The milestone 2/3 curve lives at the run root. Measuring the container
    must not be able to destroy it from the command line."""
    from splatpipe.cli import main
    from tests.test_bench_cpu import stub_views

    paths = a_run(tmp_path, n=300)
    stub_views(monkeypatch)
    sentinel = '{"sentinel": true}'
    (paths.root / "curve.json").write_text(sentinel, encoding="utf-8")

    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--codecs",
                "ply,container",
                "--output-namespace",
                "container",
            ]
        )
        == 0
    )
    assert (paths.root / "curve.json").read_text(encoding="utf-8") == sentinel
    assert paths.container_curve_json.is_file()


def test_a_baseline_codebook_round_trips_out_of_its_artifact_directory(tmp_path):
    """The container reuses the PNG baseline's own codebook so the two are
    measured on identical SH. Only the bounds and centroids come from disk;
    the labels do not, because the baseline stores them PLAS-sorted."""
    from splatpipe.compress.sh import load_sh_codebook

    rng = np.random.default_rng(0)
    centroids = rng.integers(0, 64, (32, 45), dtype=np.uint8)
    mins = rng.uniform(-1, 0, 45).astype(np.float32)
    maxs = rng.uniform(0, 1, 45).astype(np.float32)
    np.savez_compressed(
        tmp_path / "shN.npz", centroids=centroids, labels=np.zeros(4, dtype=np.uint8)
    )
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "shN": {
                    "mins": mins.tolist(),
                    "maxs": maxs.tolist(),
                    "quantization": 6,
                    "n_clusters": 32,
                }
            }
        ),
        encoding="utf-8",
    )

    stored, read_mins, read_maxs, bits = load_sh_codebook(tmp_path)
    assert bits == 6
    np.testing.assert_array_equal(stored, centroids)
    np.testing.assert_allclose(read_mins, mins)
    np.testing.assert_allclose(read_maxs, maxs)


def test_a_directory_without_an_sh_block_is_rejected(tmp_path):
    from splatpipe.compress.sh import load_sh_codebook

    np.savez_compressed(tmp_path / "shN.npz", centroids=np.zeros((2, 3), np.uint8))
    (tmp_path / "meta.json").write_text('{"means": {}}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="not an shvq run"):
        load_sh_codebook(tmp_path)
