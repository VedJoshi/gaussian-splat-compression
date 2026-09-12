"""The real thing on a synthetic scene. Needs a GPU and env.bat.

Trains for 200 steps and benchmarks all three codecs. The reconstruction is
poor and that is fine: the point is that every stage hands its output to the
next one on real pixels, not that the numbers are good.
"""

from __future__ import annotations

import pytest

from splatpipe.bench.curve import Curve
from splatpipe.cli import main
from splatpipe.paths import RunPaths
from tests.fixtures.tiny_scene import make_tiny_scene

pytestmark = pytest.mark.gpu


def test_bench_measures_every_codec_on_a_trained_scene(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    config = tmp_path / "tiny.toml"
    # cap_max is an MCMC cap rather than an exact count, so the final Gaussian
    # count is not guaranteed square and PngCompression may crop a few. The
    # assertions below do not depend on it.
    config.write_text(
        'name = "tiny"\n\n[train]\nmax_steps = 200\ncap_max = 4096\ntest_every = 8\n',
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    assert main(["run", str(scene), "--config", str(config), "--out", str(out_root)]) == 0

    paths = RunPaths.for_run(out_root, "tiny")
    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(scene),
                "--codecs",
                "ply,splat,container",
            ]
        )
        == 0
    )

    curve = Curve.read_json(paths.curve_json)
    assert [point.codec for point in curve.points] == ["ply", "splat", "container"]
    assert curve.held_out_views == 3
    assert paths.curve_png.is_file()

    ply, splat, container = curve.points
    assert container.gaussians == ply.gaussians
    assert container.bytes < ply.bytes
    assert ply.ratio == pytest.approx(1.0)
    # The .splat discards f_rest, so it is smaller than the ply and worse than it.
    assert splat.bytes < ply.bytes
    assert splat.psnr < ply.psnr
    # Every point must carry finite metrics; a nan here means a decoder clamp
    # was missed and the whole curve is meaningless.
    for point in curve.points:
        assert point.psnr == point.psnr
        assert point.gaussians > 0

    assert (
        main(
            [
                "prune",
                str(paths.root),
                "--scene",
                str(scene),
                "--retain",
                "50",
                "--codecs",
                "ply",
            ]
        )
        == 0
    )
    prune_curve = Curve.read_json(paths.prune_curve_json)
    assert [point.codec for point in prune_curve.points] == [
        "ply",
        "prune50-ply",
        "opacity50-ply",
    ]
    assert prune_curve.points[1].gaussians < prune_curve.points[0].gaussians
