from __future__ import annotations

from scripts.site_figures import plot_curve


def test_the_curve_is_an_svg_with_its_labels_as_text(tmp_path):
    points = [
        {"codec": "ply", "label": "PLY", "group": "Baselines", "bytes": 236_001_478, "psnr": 24.400},
        {"codec": "shvq4096", "label": "4,096", "group": "SH quantisation", "bytes": 14_526_458, "psnr": 24.251},
        {"codec": "prune80-container", "label": ".splatc (deployed)", "group": "Deployed .splatc", "bytes": 12_278_198, "psnr": 24.258},
    ]
    plot_curve(points, tmp_path / "curve.svg")
    svg = (tmp_path / "curve.svg").read_text(encoding="utf-8")
    assert "<svg" in svg
    # Text kept as text, so the page font applies and the label is searchable.
    assert ".splatc (deployed)" in svg
