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
